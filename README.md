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
## 实验&评估
本研究的系統架構如下，可分成兩部分來看。第一部分是訓練不含情感標籤元素的傳統生成模型，以下簡稱M1。第二部分是本研究提出的情感型生成模型，以下簡稱M2。M1是對照組，M2是實驗組，實驗組是由五種不同情感標籤的資料集分別訓練的，兩種模型的本質都是Seq2Seq神經網路。透過M1和M2，會對同樣的問句產生兩種答句，再由人工方式，對抽樣後的兩種答句進行評估。

#### 论文系统架构图
* ![image](https://github.com/a362758199/STC3-CECG/blob/master/charts/%E4%B8%AD%E4%B8%80%E8%AB%96%E6%96%87%E7%B3%BB%E7%B5%B1%E6%9E%B6%E6%A7%8B%E5%9C%96.png) 

#### 实验结果对比图 
* ![image](https://github.com/a362758199/STC3-CECG/blob/master/charts/问卷實驗結果對比圖.png)

## 深度学习模型
### Word Embedding
* ![image](https://github.com/a362758199/STC3-CECG/blob/master/charts/%E4%BD%99%E5%BC%A6%E7%9B%B8%E4%BC%BC%E5%BA%A6.png)
### GRU with attention generator
* ![image](https://github.com/a362758199/STC3-CECG/blob/master/charts/attention-decoder-network.png)
### CNN sentiment classifier
* ![image](https://github.com/a362758199/STC3-CECG/blob/master/charts/CNN%20classification.png)

## 深度学习模型的包装
#### Flask app网页介面展示
* ![image](https://github.com/a362758199/STC3-CECG/blob/master/charts/%E8%9E%A2%E5%B9%95%E5%BF%AB%E7%85%A7_2019-08-27_03-39-08.png)
