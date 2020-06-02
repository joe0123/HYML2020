## 學號：B06902125 系級：資工三 姓名：黃柏瑋 

#### 1.(3%)

###### 請至少使用兩種方法 (autoencoder 架構、optimizer、data preprocessing、後續降維方法、clustering 算法等等) 來改進 baseline code 的 accuracy。分別記錄改進前、後的 test accuracy 為多少。分別使用改進前、後的方法，將 **val data** 的降維結果 (embedding) 與他們對應的 label 畫出來。盡量詳細說明你做了哪些改進。

首先，先對autoencoder的訓練下手。為了讓autoencoder產生的embedding更general一些，在訓練autoencoder前，隨機為每張照片進行以下其中一種data augmentation技巧，將dataset擴增為兩倍大：

1. 隨機旋轉照片30到45度
2. 水平翻轉

並在相同的模型之下，將learning rate從1e-5調大至5e-4，同樣以batch size 64訓練100個epoch。

接著，改進kmeans的算法。有些時候，看起來屬於同一群的的兩個點，可能會因為與他們最近的centroid不同而被分到不同的cluster之中。然而，他們應該要被分到同一類比較合適。因此，先將所有的資料分為64個cluster，再將這64個centroid分成2個cluster也許會比較合適一些。示意圖如下：

(Kmeans: 2 clusters)

![scatter_1](D:\files\College life (Junior)\Second Semester\HYML\HW9\scatter_1.jpg)

(Kmeans: 64 clusters -> Kmeans: 2 clusters)

![scatter_2](D:\files\College life (Junior)\Second Semester\HYML\HW9\scatter_2.jpg)

以下為改進前與改進後的test accuracy比較：

|               | before | after                           |
| ------------- | ------ | ------------------------------- |
| Test accuracy | 0.7115 | <font color='red'>0.7798</font> |

以下分別為改進前與改進後的embedding plots。可以發現改進之後，不同類的embedding分布有比較分離。

(Before)

![scatter_basic](D:\files\College life (Junior)\Second Semester\HYML\HW9\scatter_basic.jpg)

(After)

![scatter_improved2](D:\files\College life (Junior)\Second Semester\HYML\HW9\scatter_improved.jpg)

#### 2. (1%)

###### 使用你 test accuracy 最高的 autoencoder，從 trainX 中，取出 index 1, 2, 3, 6, 7, 9 這 6 張圖片畫出他們的原圖以及 reconstruct 之後的圖片。

![reconst](D:\files\College life (Junior)\Second Semester\HYML\HW9\reconst.jpg)

#### 3. (2%)

###### 在 autoencoder 的訓練過程中，至少挑選 10 個 checkpoints 請用 model 的 train reconstruction error (用所有的 trainX 計算 MSE) 和 **val accuracy** 對那些 checkpoints 作圖。簡單說明你觀察到的現象。

以下結果由problem 1中進步後的模型產生：

![Q3](D:\files\College life (Junior)\Second Semester\HYML\HW9\Q3.jpg)

不難發現，其實autoencoder的loss越低，並不保證accuracy越高，而從以下的embedding plot中也可以發現，拿checkpoint 100去做clustering的效果比checkpoint 40的還要差。可能的原因是當autoencoder學得太好時，同時也學到太多圖片中的雜訊，進而影響clustering的效果。

(checkpoint 40)

![scatter_true100](D:\files\College life (Junior)\Second Semester\HYML\HW9\scatter_true40.jpg)

(checkpoint 100)

![scatter_true40](D:\files\College life (Junior)\Second Semester\HYML\HW9\scatter_true100.jpg)