# HDDAS：幽默識別之資料增強策略選擇


## Introduction
對機器學習而言，基本上來說愈多資料愈能有效提升perfomance，資料增強（Data Augmentation, DA)，在電腦視覺（Computer Vision, CV)領域相當實用，例如將影像旋轉、鏡像或甚至白化等多種操作。在自然語言處理（Nature Language Processing, NLP)領域，因為語言的複雜性，若是隨意替換文字，很可能就造成語意上的歧義，因此相較於CV，NLP要將資料增強稍嫌困難。

幽默多指令人發笑的品質或者具有發笑的能力，它是一種特殊的語言表達方式，是生活中活躍氣氛、化解尷尬的重要元素。近年來隨著人工智慧的快速發展，如何利用計算機技術識別和生成幽默逐漸成為NLP領域研究熱點之一，即「幽默計算」。「幽默計算」旨在賦予計算機識別、生成幽默的能力，它涉及資訊科學、認知語言學、心理學等多個學科的交叉，在人類語言的理解乃至世界文化的交流方面，都具有重要的理論和應用價值。

此Project專注在幽默計算的其中一項任務：幽默識別（Humor Detection, HD)，也就是判别某一句子是否包含一定程度的幽默，並試著對也屬於文字分類任務的HD使用DA，且根據HD資料集的特性，嘗試設計最適合的DA策略，以求提升在這個任務的perfomance。

## Related Work
在幽默識別方面，非神經網路模型(non-NN-based)可以[Yang et al. (ACL-15)](https://www.aclweb.org/anthology/D15-1284.pdf)為代表，其使用Feature Engineering的方式，考慮到幽默的latent semantic structures，針對Incongruity、Ambiguity、Interpersonal Effect和Phonetic Style等面向，從句子產生feature再使用Random Forest，也提供了資料集：Pun of the day；NN-based 可以[Chen and Soo (ACL-18)](https://www.aclweb.org/anthology/N18-2018.pdf)為代表，其主要使用卷積神經模型（Convolutional Neural Network, CNN)，搭配Highway機制有效提升CNN的performance；[Weller and Seppi (EMNLP-IJCNLP-19)](https://www.aclweb.org/anthology/D19-1372.pdf)嘗試在此任務中使用NLP State-of-art的transformer-based模型[BERT](https://arxiv.org/abs/1810.04805)，performance在各個資料集皆有相當的成長。

在資料增強方面，[Wei and Zou (ACL-19)](https://www.aclweb.org/anthology/D19-1670.pdf)提出Easy Data Augmentation(EDA)，使用Synonym Replacement、Random Insertion、Random Swap和Random Deletion，簡單且有效提升文字分類任務的performance；[Xie et al. (2019)](https://arxiv.org/abs/1904.12848)提出Unsupervised Data Augmentation(UDA)，使用Back translation和TF-IDF word replacement等技巧將大量資料標記。

[Yang et al. (AAAI-20)](https://rcqa-ws.github.io/papers/paper3.pdf) 將資料增強應用在幽默識別上，同樣以BERT作為模型，針對資料集和BERT的特性，設計Paragraph Decomposition，有效提升資料大小及performance。

## Dataset
[Pun of the day](https://github.com/orionw/RedditHumorDetection/blob/master/full_datasets/puns/puns_pos_neg_data.csv)，由[Yang et al. (ACL-15)](https://www.aclweb.org/anthology/D15-1284.pdf)所蒐集並開源，共有4222筆資料，每筆資料為最多不到65個字的短句，Positive Sample也就是含有幽默的句子佔2116筆，主要來自[punoftheday 網站](https://www.punoftheday.com/)；Negative Sample也就是不含有幽默的句子佔2106筆，主要來自AP News、New York Times、Yahoo! Answer 和 Proverb等網站。

### 資料一覽

| dataset | Positive | Negative | Total | Percent |
| ------- | -------- | -------- | ----- | ------- |
| train   | 1809     | 1810     | 3619  | 85%     |
| test    | 307      | 296      | 603   | 15%     |
| -       | -        | -        | 4222  | 100%    |

Positive Example
> an elevator makes ghosts happy because it lifts the spirits


Negative Example
> I m hoping they ll come and see this and say We have to have this


## Approach
此Project 採用BERT 作為主要的模型，使用[Huggingface](https://github.com/huggingface/transformers) naive setting 的pretrained model `bert-base-uncased`，接上`BertForSequenceClassification` finetune，再搭配以下資料增強策略：

### 1. Paragraph Decomposition (PD)
根據[幽默語義腳本理論 (Script-based Semantic Theory of Humor , SSTH)](https://www.researchgate.net/publication/273946710_Semantic_Mechanisms_of_Humor)，將一段幽默的句子或稱之為笑話的結構定義為由`主體` (`set-up`)和`妙語` (`punch line`)兩部分組成，主體至少有兩個可能的解釋(即兩個腳本)，其中只有一個解釋是顯而易見的，而幽默效果則由觸發第二個不明顯解釋的妙語所產生。
以下列句子為例：
> i used to be a banker but i lost interest

`set-up` 為 
> i used to be a banker

而`punch line`則是
> but i lost interest


[Yang et al. (AAAI-20)](https://rcqa-ws.github.io/papers/paper3.pdf) 考慮到幽默識別資料集的特性，也就是如上一段所說，基本上句子由`set-up`和`punch line`組成，且觀察到順序通常是先`set-up`再`punch line`。並考慮BERT的特性，在train的時候須完成Next Sentence任務，且會有兩個特殊token：`[CLS]`&`[SEP]`。

以下列句子為例說明
```
貓似乎只是在削尖他们的爪子。實際上，他们正在鍛鍊腿部肌肉。
```
通常在做單一句子分類時，會改成以下作BERT的input：
```
[CLS]貓似乎只是在削尖他们的爪子。實際上，他们正在鍛鍊腿部肌肉。[SEP]
```
由以上考慮，他們會抓出句子中所有的標點符號，在其後加上`[SEP]`，
```
[CLS]貓似乎只是在削尖他们的爪子。[SEP]實際上，他们正在鍛鍊腿部肌肉。[SEP]
[CLS]貓似乎只是在削尖他们的爪子。實際上，[SEP]他们正在鍛鍊腿部肌肉。[SEP]
```

藉由此decomposition的操作，對BERT來說是不同但卻不會影響context的input，以達到data augmentation的效果，且希望藉此自動抓出`punch line`的位置。

受其啟發，此final project也將句子decompose成兩個部分，但由於Pun of the day並無標點符號且為英文，故直接在句中插入`[SEP]`，且不在第一個字和第二個字中間的位置。同樣以Positive example作範例：
```
[CLS] an elevator [SEP] makes ghosts happy because it lifts the spirits [SEP]
[CLS] an elevator makes [SEP] ghosts happy because it lifts the spirits [SEP]
[CLS] an elevator makes ghosts [SEP] happy because it lifts the spirits [SEP]
[CLS] an elevator makes ghosts happy [SEP] because it lifts the spirits [SEP]
[CLS] an elevator makes ghosts happy because [SEP] it lifts the spirits [SEP]
[CLS] an elevator makes ghosts happy because it [SEP] lifts the spirits [SEP]
[CLS] an elevator makes ghosts happy because it lifts [SEP] the spirits [SEP]
[CLS] an elevator makes ghosts happy because it lifts the [SEP] spirits [SEP]
```

### 2. Paragraph Swap (PS)
[Wei and Zou (ACL-19)](https://www.aclweb.org/anthology/D19-1670.pdf)所提出的EDA中，其中有一項是 Random Swap，即是把不同句子互相crossover，但由於其原本使用的任務是情感分析，若是極性相同，不同句子合併極性不會有太大改變，但此任務是幽默識別，將句子Swap將破壞`set-up` 和`punch line`的結構。

受其啟發，若是能找出原句的`set-up` 和`punch line`分別為何，再將其前後Swap，既不會破壞句子context也不會造成太大歧義。
而至於如何找出`punch line`，則藉由找在PD階段`[SEP]`放在哪一個位置時，model所output為positive的機率最大，也就是將其視為分別`set-up`和`punch line`達到最大幽默效果的位置。
同樣以Positive example作範例，以下位置的機率最大：
```
[CLS] an elevator makes ghosts happy because it lifts [SEP] the spirits [SEP]
```
PS後則變成
```
[CLS] the spirits [SEP] an elevator makes ghosts happy because it lifts [SEP]
```

### 3. Synonym Replacement (SR)
同樣[Wei and Zou (ACL-19)](https://www.aclweb.org/anthology/D19-1670.pdf)所提出的EDA中，除了第二點的Random Swap，還有一項是Synonym Replacement，受其啟發，這邊基本上與之同樣作法，藉由wordNet找出sense相似或甚至相同的替換。而考慮到`punch line`還是造成幽默的關鍵，且為了避免換掉當中重要的部分，這邊SR只會作`set-up`的部分，且多生成5句。同樣以Positive example作範例：
```
[CLS] an lift makes ghosts happy because information technology lifts [SEP] the spirits [SEP]
[CLS] an elevator makes ghosts happy because it lift [SEP] the spirits [SEP]
[CLS] an elevator makes specter happy because it lifts [SEP] the spirits [SEP]
[CLS] an elevator makes ghostwriter happy because it lifts [SEP] the spirits [SEP]
[CLS] an elevator makes ghosts happy because information technology lifts [SEP] the spirits [SEP]
```

## Experiments

| Method              | Accuracy |
| ------------------- | -------- |
| BERT                | 0.9104   |
| BERT + PD           | 0.9137   |
| BERT + PD + PS      | 0.9170   |
| BERT + PD + SR      | 0.9237   |
| BERT + PD + SR + PS | 0.9237   |

模型設定基本上都是一樣，主要探討不同data augmentation的策略會有什麼不同的影響。最後皆取Epoch為2的checkpoint，再多的epoch就有overfitting的趨勢。
從表格可以看到，baseline即有一定的水準，透過PD後，有些微的成長，此後基於PD再加上其他data augmentation的方法，performance也都有些許上升。

## Conclusion & Future Work
- 在幽默識別任務下，Paragraph Decomposition (PD)、Paragraph Swap (PS)和Synonym Replacement (SR) 等資料增強的策略皆可使模型表現有稍許的提升。

- Word Embedding SR：同Approach第三點的概念，換成用如`word2vec`、`GloVe`和`fasttext`等static word embedding 找相似字做替換，以及使用context word embedding如`ELMO`、`BERT`作替換。

## References
- [Humor Recognition and Humor Anchor Extraction (ACL-15)](https://www.aclweb.org/anthology/D15-1284.pdf)
- [Humor recognition using deep learning (ACL-18)](https://www.aclweb.org/anthology/N18-2018.pdf)
- [Humor Detection: A Transformer Gets the Last Laugh (EMNLP-IJCNLP-19)](https://www.aclweb.org/anthology/D19-1372.pdf)
- [EDA: Easy Data Augmentation Techniques for Boosting Performance on Text Classification Tasks (ACL-19)](https://www.aclweb.org/anthology/D19-1670.pdf)
- [Unsupervised Data Augmentation for Consistency Training (2019)](https://arxiv.org/abs/1904.12848)
- [Humor Detection based on Paragraph Decomposition and BERT Fine-Tuning (AAAI-20 RCQA workshop)](https://rcqa-ws.github.io/papers/paper3.pdf)
- [Semantic Mechanisms of Humor (1985)](https://www.researchgate.net/publication/273946710_Semantic_Mechanisms_of_Humor)
