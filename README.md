# YY

This code is proposed on "**Extension and Comparison of Deterministic Black-box Adversarial Attacks**" in [SCIS2020](https://www.iwsec.org/scis/2020/).

- examples
<div align="center">
  <img src="https://user-images.githubusercontent.com/60645850/73763102-512fe480-47b4-11ea-94a5-e01ef4ff6847.png" width="500px">
</div>

- YY generates cross label adversarial on CIFAR-10. Labels on the left are the true labels, labels on the bottom are predicted labels by the model.
<div align="center">
  <img src="https://user-images.githubusercontent.com/60645850/73920006-f1e2e900-4907-11ea-83e7-06aaa2ec1ee0.png" width="800px">
</div>



## Dependencies

1. Python3
2. Numpy
3. Matplotlib
4. Tensorflow

Sample codes require these libraries. Please install them on your environment.



## How to Use

- train MNIST model
  ```python
  $ python train.py -d mnist -e 10 -m cnn_1
  ```

- train CIFAR-10 model
  ```python
  $ python train.py -d cifar10 -e 50 -m cnn_2
  ```
  
- targeted attack on MNIST
  ```python
  $ python attack.py -d mnist -e 0.3 -m cnn_1 -t
  ```
  If you delete -t option, attack on untargeted.

- targeted attack on CIFAR-10
  ```python
  $ python attack.py -d cifar10 -e 0.05 -m cnn_2 -t
  ```
  If you delete -t option, attack on untargeted.



## Notice

```python
print('\nExcluding misclassification samples')
(X_test, y_test) = exclude_miss(sess, env, X_test, y_test, 0, 10)
evaluate(sess, env, X_test, y_test)
```

```exclude_miss()``` in attack.py excludes samples originally misclassified by trained model in ```0-9``` pages of ```X_test``` datasets. If you run with more data, increase the number of the part of ```(0, 10)``` by yourself.




## License

Some of the codes including this source use [gongzhitaao/tensorflow-adversarial](https://github.com/gongzhitaao/tensorflow-adversarial). Thanks.

