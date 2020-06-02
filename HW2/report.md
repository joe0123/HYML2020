## 學號：B06902125 系級：資工三 姓名：黃柏瑋

### 1.(2%)

**請比較實作的 generative model 及 logistic regression 的準確率，何者較佳？請解釋為何有這種情況？**

在兩者x和y一樣的前提下，成績如下：

|             | generative model | logistic regression |
| ----------- | ---------------- | ------------------- |
| training    | 0.8745           | 0.8820              |
| public test | 0.8836           | 0.8883              |

由上可知，logistic regression在預測的表現上優於generative model。這樣的情形可能是因為generative model認為資料來自機率模型，為資料添了一些假設，適用於資料量少或雜訊高的題目；而此次作業提供的資料量相當充足，利用discriminative model(如logistic regression)能達到更好的分類結果。

### 2. (2%)

**請實作 logistic regression 的正規化 (regularization)，並討論其對於你的模型準確率的影響。接著嘗試對正規項使用不同的權重 (lambda)，並討論其影響。**

實作和講義一樣的L2-regularization：

(without regularization)
![image-20200323221237229](C:\Users\user\AppData\Roaming\Typora\typora-user-images\image-20200323221237229.png)

(with L2-regularization, λ=1e-2)
![image-20200323221130745](C:\Users\user\AppData\Roaming\Typora\typora-user-images\image-20200323221130745.png)

由於Logistic regression模型簡單，訓練的時候沒有嚴重的overfitting產生，加入regularization後對於validation data的準確率影響不大，但如果多著眼於loss的話，可以發現有加regularization的training和valid loss會稍微降低，代表regularization還是有稍微優化模型。

再嘗試一些不同的λ：

(with L2-regularization, λ=1e-1)
![image-20200323222430219](C:\Users\user\AppData\Roaming\Typora\typora-user-images\image-20200323222430219.png)

當λ=1e-1時，訓練的成效不佳，出現了underfitting的現象，在training和valid的表現都比不加regularization還差。

(with L2-regularization, λ=2e-2)
![image-20200323223436982](C:\Users\user\AppData\Roaming\Typora\typora-user-images\image-20200323223436982.png)

當λ=2e-2時，準確率和λ=1e-2時差不多，但loss方面又更降低許多，效果更好一些。

(with L2-regularization, λ=1e-3)![image-20200323222911461](C:\Users\user\AppData\Roaming\Typora\typora-user-images\image-20200323222911461.png)

當λ=1e-3時，regularization對模型的penalty有點太小，成效不彰，結果和沒有加regularization的差不多。

總而言之，regularization在validation準確率上影響不算大，而λ在1e-2附近的效果會比較好。

### 3. (1%)

**請說明你實作的 best model，其訓練方式和準確率為何？**

我的best model主要是建立在basic的logistic regression之上，僅對input做了feature engineering而已。

首先先把training data中完全沒出現過的feature移除："other rel < 18 ever marr not in subfamily"和"grandchild < 18 never marr rp of subfamily"。

接著對連續數值的feature進行分群(binning)，分群原則是將鄰近且分布相近的區間歸在同一群，舉"age"為例：0到10歲與10到18歲大約都只有0.5%以下的人年薪大於50000，而18到25歲大約有2%，因此"age"的第一群便由0到18歲的人組成。
分群結果如下：

| feature                         | bin boundaries (a, B]                   |
| ------------------------------- | --------------------------------------- |
| age                             | -1, 18, 25, 35, 45, 55, 65, 75, np.inf  |
| capital gains                   | -np.inf, 4600, 7600, 15000, np.inf      |
| capital losses                  | -np.inf, 1400, 2000, 2200, 3200, np.inf |
| dividends                       | -np.inf, 0, 5000, np.inf                |
| num persons worked for employer | -1, 0, 1, 2, 3, 4, 5, 6                 |
| working weeks                   | -np.inf, 25, 45, np.inf                 |
| wage per hour                   | -np.inf, 0, 1200, 1800, 2200, np.inf    |

此外，我也用Keras架設Deep Neural Network，試了一些不同的架構，而最後用了以下的模型架構：

![image-20200325173828368](C:\Users\user\AppData\Roaming\Typora\typora-user-images\image-20200325173828368.png)

接著我利用logistic regression with L2-regularization挑掉權重小於0.1的feature，又因為data中label=0比label=1的多，於是在訓練模型時給上class weight(1:1.2)，平衡在計算loss時倒向label=0的窘境。

以下表格為逐步優化模型時的準確率：

| Model                                                        | valid acc | Public test Acc |
| ------------------------------------------------------------ | --------- | --------------- |
| Logistic Regression <br />with feature binning               | 0.8869    | 0.8943          |
| DNN<br />with feature binning                                | 0.8879    | 0.8953          |
| DNN<br />with feature binning, selection<br />and class weight | 0.8884    | 0.8958          |



### 4. (1%)

**請實作輸入特徵標準化 (feature normalization)，並比較是否應用此技巧，會對於你的模型有何影響。**

在logistic regression的basic models中，若不用normalization會使訓練過程中的loss及accuracy不穩定地跳動，也不好收斂(如下圖所示)；若使用standardization或min-max normalization可以改善這個問題，其中standardization的表現較佳。

(without normalization)
![image-20200323191821504](C:\Users\user\AppData\Roaming\Typora\typora-user-images\image-20200323191821504.png)

(min-max)![image-20200323192105743](C:\Users\user\AppData\Roaming\Typora\typora-user-images\image-20200323192105743.png)

(standardization)
![image-20200323192651858](C:\Users\user\AppData\Roaming\Typora\typora-user-images\image-20200323192651858.png)

在best model中，由於所有的feature都是one-hot形式，因此沒有使用nomalization(也可以說使用min-max)的效果最好。

(min-max)
![image-20200323193549992](C:\Users\user\AppData\Roaming\Typora\typora-user-images\image-20200323193549992.png)

(standardization)
![image-20200323193201692](C:\Users\user\AppData\Roaming\Typora\typora-user-images\image-20200323193201692.png)