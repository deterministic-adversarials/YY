import numpy as np
import tensorflow as tf


__all__ = [
    'YY'
]


## xのラベルとtargetに対する確率を返す
def pred(sess, env, x, target):
    ybar = sess.run(env.ybar, feed_dict={env.x: x})
    ybar = np.squeeze(ybar)
    y = np.argmax(ybar)
    proba = ybar[target]
    return y, proba


##
def YY(sess, env, x, target, eps, img_size, img_chan):
    
    num_queries = 0

    for split in range(1,img_size+1):
        xadv = np.copy(x)  

        ## 画像の縦(横)を分割数splitで等分
        # 例 : 28を3等分 -> interval[0]=10, interval[1]=9, interval[2]=9
        interval = np.zeros((split), dtype=int)
        for i in range(img_size):
            j = interval[i%split]
            interval[i%split] = j + 1

        ## split * split * img_chan で分割しノイズ付加
        for c in range(img_chan):
            h_start = 0
            h_end = 0
            for height in range(split):
                h_start = h_end
                h_end = h_end + interval[height]
                w_start = 0
                w_end = 0
                for width in range(split):
                    w_start = w_end
                    w_end = w_end + interval[width]
                    
                    ## クエリ
                    # prob = ノイズを付加する前のtargetに対する確信度
                    # yadv = モデルのxadvに対する推測クラス
                    yadv, prob = pred(sess, env, xadv, target)
                    num_queries += 1

                    ## 攻撃が成功していた場合はreturn
                    if env.targeted:
                        if yadv == target:
                            return xadv, num_queries, split, True
                    else:
                        if yadv != target:
                            return xadv, num_queries, split, True
                    
                    ## ノイズ
                    mat_eps = np.zeros((img_size,img_size,img_chan), dtype=float)
                    mat_eps[h_start:h_end,w_start:w_end,c] = eps
                    mat_eps = np.stack([mat_eps])

                    ## ノイズ付加
                    xadv_plus = xadv + mat_eps
                    xadv_minus = xadv - mat_eps
                    # クリッピング
                    xadv_plus = np.clip(xadv_plus, 0., 1.)
                    xadv_minus = np.clip(xadv_minus, 0., 1.)
                    
                    ## クエリ
                    # prob_plus = ノイズ(+eps)を付加した後のtargetに対する確信度
                    # yadv = モデルのxadv_plusに対する推測クラス
                    yadv, prob_plus = pred(sess, env, xadv_plus, target)
                    num_queries += 1

                    ## 攻撃が成功していた場合はreturn
                    if env.targeted:
                        if yadv == target:
                            return xadv_plus, num_queries, split, True
                    else:
                        if yadv != target:
                            return xadv_plus, num_queries, split, True

                    ## 確信度の比較 -> xadv更新
                    # targeted攻撃のとき
                    if env.targeted:
                        # 確信度が上昇
                        if prob < prob_plus:
                            xadv = xadv_plus
                        else:
                            xadv = xadv_minus
                    # untargeted攻撃のとき
                    else:
                        # 確信度が下降
                        if prob >= prob_plus:
                            xadv = xadv_plus
                        else:
                            xadv = xadv_minus
    
    return xadv, num_queries, split, False
