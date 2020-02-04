# YY

[SCIS2020](https://www.iwsec.org/scis/2020/) 3D2-1「Black-box攻撃における決定的Adversarial Examples生成手法の拡張と比較」にて提案した手法。

<div align="center">
  <img src="https://user-images.githubusercontent.com/60645850/73763102-512fe480-47b4-11ea-94a5-e01ef4ff6847.png" width="500px">
</div>



## 使用するライブラリ

- numpy
- matplotlib
- tensorflow



## 使い方

- モデルの訓練(MNIST)

  ```python
  $ python train_model_for_mnist.py
  ```
  
- 攻撃(MNIST)

  ```python
  $ python attack_on_mnist.py
  ```

CIFAR-10についても同様。



## ライセンスについて

YYに含まれるソースコードの一部は [gongzhitaao/tensorflow-adversarial](https://github.com/gongzhitaao/tensorflow-adversarial) を使わせていただいています。
