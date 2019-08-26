# STC3-CECG
* [論文全文pdf](https://github.com/a362758199/STC3-CECG/blob/master/%E9%9B%A2%E6%A0%A1%E7%89%88%E8%AB%96%E6%96%87%E6%9C%AC%E6%96%87627.pdf)
* [conference proceeding of TKUIM](https://github.com/a362758199/STC3-CECG/blob/master/NTCIR-14_paper_27_v6.pdf)
* [poster of TKUIM](https://github.com/a362758199/STC3-CECG/blob/master/NTCIR14-STC-TKUIM-Wei%20Shih%20chieh-poster.pdf)
* [情感型聊天机器人调查问卷【填寫入口】](https://docs.google.com/forms/d/e/1FAIpQLSc1oNbpzHLD4xNwdjNIHjc63ZCseXR9jTyYFLQfcycAfscXiA/viewform?usp=sf_link)
## NTCIR
NTCIR是一個針對資訊架構（Information Architecture，IA）技術的競賽，其中包含問題回答，資訊檢索，資訊萃取和文本摘要等。NTCIR最早由日本國立情報學研究所（National Institute of Information, NII），前身為國立學術情報中心（National Center for Science Information System, NACSIS）和日本學術振興會（Japan Society for the Promotion of Science, JSPS）聯合贊助，於1998年開始籌備，並在1999年成功舉辦首屆工作坊。經過二十年發展NTCIR已成為一項國際重要賽事，曾設置了一系列中文，日文，英文等亞洲語言的評估任務，目前舉辦至第十四屆。
	
第十四屆NTCIR的STC競賽中，依然有中文評測任務。其中，短文對話任務在上一屆引入深度學習對話生成模型之後，這一屆又增加了情感標籤的元素，希望達成基於情感標籤的對話生成任務（Chinese Emotional Conversation Generation，CECG）。本研究團隊有幸參加了此次NTCIR 14-STC3的CECG子任務。也正是本次比賽啟發本文使用情感標籤分類，作為提升句子生成品質的一種方法。
#### 资料集预览图
* ![image](https://github.com/a362758199/STC3-CECG/blob/master/charts/%E8%B5%84%E6%96%99%E9%9B%86%E9%A2%84%E8%A7%88.PNG)

## 深度学习模型

本研究中使用到的是一種分散式詞表示法。該方法以深度學習為基礎，將文本中的每個詞訓練成不同的短向量，并將它們集中在一個向量空間中。在這個空間里有距離的概念，如餘弦相似度（Cosine similarity）。這樣一來，就可以用詞之間的距離表示相關度。
最常用的詞向量訓練法有兩種：連續詞袋法（continuous bag of words, CBOW）和跳躍詞法（Skip-Gram）。兩者的原理類似，實現的方式相反，可參考相關文獻。

### Word Embedding
* ![image](https://github.com/a362758199/STC3-CECG/blob/master/charts/%E4%BD%99%E5%BC%A6%E7%9B%B8%E4%BC%BC%E5%BA%A6.png)

Seq2Seq 可以簡單看做是一個由編碼器及解碼器（Encoder–Decoder）兩個RNN結構組成的網路。其輸入是一個序列，輸出也是一個序列。編碼器的作用是將一個可變長度的序列轉成固定長度的情境向量（context vector）。而解碼器則將這個固定長度的情境向量c變成可變長度的目標序列。下圖問本研究中用到的解碼器類型，加入了一個Att模塊。

### GRU with attention generator
* ![image](https://github.com/a362758199/STC3-CECG/blob/master/charts/attention-decoder-network.png)

Kim.Y在2014年提出一種使用CNN對句子進行分類的方法。卷積神經網路在影像處理中，使用不同的濾鏡來使圖像凸顯不同的特徵，文字也是一樣。圖8中CNN的第一層原始矩陣在通過濾鏡之後會得到第二層的矩陣，我們稱之為特徵圖。特徵圖通過第三層池化層之後，會得到更小的特徵圖，就像人眼在觀察圖片時首先觀察到的那些耀眼部分。在這個文字分類器中，我們所需要學習的部分就是這個濾鏡，也就是CNN中的卷積核。
由左至右，第一層輸入層是將一個句子所有單詞的詞向量進行拼接的矩陣，每一橫行代表一個詞。第二層卷積層，每個卷積核的大小為filter_size * embedding_size，filter_size代表卷積核縱向上一次要看的單詞個數，即認為相鄰幾個詞之間有詞序關係，embedding_size就是詞向量的維度。每個卷積核計算完成之後就得到1個行向量，代表著該卷積核從句子中提取出來的濾鏡特徵，有多少個卷積核就能提取出多少種濾鏡特徵。第三層池化層的操作就是將卷積得到的豎行向量的最大值提取出來，通過池化之後會獲得一個維度等於卷積核數量的橫列向量，即將每個卷積核的最大值連接起來。最後一層全連接層，為了得到預期的結果，將池化層的輸出向量透過一個softmax，得到兩個和為1的數值，在本文中，即是該句子屬於該分類和不屬於該分類的機率值。

### CNN sentiment classifier
* ![image](https://github.com/a362758199/STC3-CECG/blob/master/charts/CNN%20classification.png)

## 实验&评估
本研究的系統架構如下，可分成兩部分來看。第一部分是訓練不含情感標籤元素的傳統生成模型，以下簡稱M1。第二部分是本研究提出的情感型生成模型，以下簡稱M2。M1是對照組，M2是實驗組，實驗組是由五種不同情感標籤的資料集分別訓練的，兩種模型的本質都是Seq2Seq神經網路。透過M1和M2，會對同樣的問句產生兩種答句，再由人工方式，對抽樣後的兩種答句進行評估。

#### 论文系统架构图
* ![image](https://github.com/a362758199/STC3-CECG/blob/master/charts/%E4%B8%AD%E4%B8%80%E8%AB%96%E6%96%87%E7%B3%BB%E7%B5%B1%E6%9E%B6%E6%A7%8B%E5%9C%96.png) 

實驗從三個角度評估生成句的質量，權重賦值相等。
* 語言流暢度：指答句在表達時的邏輯，語法是否符合常識規範
* 問答相關度：指答句與問句所表達的內容是否相關
* 情感表達度：指答句所內涵的情感表達針對問句是否合理或合適
#### 实验结果对比图 
* ![image](https://github.com/a362758199/STC3-CECG/blob/master/charts/问卷實驗結果對比圖.png)

## 深度学习模型的包装
#### Flask app网页介面展示
* ![image](https://github.com/a362758199/STC3-CECG/blob/master/charts/%E8%9E%A2%E5%B9%95%E5%BF%AB%E7%85%A7_2019-08-27_03-39-08.png)
