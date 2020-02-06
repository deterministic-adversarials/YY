# YY

[SCIS2020](https://www.iwsec.org/scis/2020/) 3D2-1「**Black-box攻撃における決定的Adversarial Examples生成手法の拡張と比較**」にて提案した手法。

- 例
<div align="center">
  <img src="https://user-images.githubusercontent.com/60645850/73763102-512fe480-47b4-11ea-94a5-e01ef4ff6847.png" width="500px">
</div>

- 10クラスそれぞれのサンプルに対しtargetに10クラス指定
<div align="center">
  <img src="https://user-images.githubusercontent.com/60645850/73920006-f1e2e900-4907-11ea-83e7-06aaa2ec1ee0.png" width="800px">
</div>



## 使用するライブラリ

- numpy
- matplotlib
- tensorflow

各環境でインストールしてください。



## 使い方

- モデルの訓練(MNIST)
  ```python
  $ python train.py -d mnist -e 10 -m cnn_1
  ```

- モデルの訓練(CIFAR-10)
  ```python
  $ python train.py -d cifar10 -e 50 -m cnn_2
  ```
  
- targeted攻撃(MNIST)
  ```python
  $ python attack.py -d mnist -e 0.3 -m cnn_1 -t
  ```
  -tオプションを消すとuntargeted攻撃

- targeted攻撃(CIFAR-10)
  ```python
  $ python attack.py -d cifar10 -e 0.05 -m cnn_2 -t
  ```
  -tオプションを消すとuntargeted攻撃



## attack.py

```python
print('\nExcluding misclassification samples')
(X_test, y_test) = exclude_miss(sess, env, X_test, y_test, 0, 10)
evaluate(sess, env, X_test, y_test)
```
exclude_miss()でX_testデータセット及びy_testデータセット0~9枚目の中でもともと誤分類してしまうサンプルを除去します。引数の数字を大きくすればその分testに使うデータ量も増えます。



## ライセンスについて

YYに含まれるソースコードの一部は [gongzhitaao/tensorflow-adversarial](https://github.com/gongzhitaao/tensorflow-adversarial) を使わせていただいています。
