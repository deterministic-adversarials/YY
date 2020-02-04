import os
import numpy as np
import tensorflow as tf
import statistics

from attacks import YY
from models import model_cifar10
from tools import evaluate, predict, pseudorandom_target, load_cifar10, out_, sample_img, exclude_miss


img_size = 32
img_chan = 3
n_classes = 10


def make_adv(sess, env, X_data, y_data):

    print('\nMaking adversarials')

    X_adv = np.empty_like(X_data)
    success_query = []
    q = np.zeros((img_size+1), dtype=np.int32)

    ## params
    eps = 0.05
    # True = targeted攻撃
    env.targeted = True

    os.makedirs('querydata', exist_ok=True)
    name_out = 'CIFAR10_'+str(eps)+'_'+str('targeted' if env.targeted else 'untargeted')+'_YY'
    path_out = './querydata/'+name_out+'.txt'

    with open(path_out, mode='w') as f:
        for idx, x in enumerate(X_data):
            print('\n batch {0}/{1}'.format(idx+1, len(X_data)))
            x = np.stack([x])

            ## ターゲット設定
            if env.targeted:
                target = np.argmax(y_data[idx])
                target = pseudorandom_target(idx, n_classes, target)
            else:
                target = np.argmax(y_data[idx])
            
            ## xに対しA.E.生成
            xadv, num_queries, split, success = YY(
                    sess, env, x, target, eps, img_size, img_chan)
            
            ## 集計
            yadv = sess.run(env.ybar, feed_dict={env.x: xadv})
            yadv = np.squeeze(yadv)
            yadv = np.argmax(yadv)
            if success:
                X_adv[idx] = xadv
                q[split] += 1
                success_query.append(num_queries)
                f.write(str(num_queries)+'\n')
                if env.targeted:
                    print('  y='+str(np.argmax(y_data[idx]))+' yadv(target)='+str(yadv)+' query='+str(num_queries))
                else:
                    print('  y='+str(np.argmax(y_data[idx]))+' yadv='+str(yadv)+' query='+str(num_queries))
            else:
                print('  y='+str(np.argmax(y_data[idx]))+' yadv='+str(yadv)+' query='+str(num_queries))
                X_adv[idx] = x
                q[0] += 1
                print('  failed')

        f.write('-----------------------------\n')
        f.write('all_sample = '+str(len(X_data))+'\n')
        f.write('success_sample = '+str(len(success_query))+'\n\n')

        f.write('cifar10_YY : eps='+str(eps)+' <'+str('targeted' if env.targeted else 'untargeted')+'>\n')
        f.write('mean:'+str(statistics.mean(success_query))+'\n')
        f.write('median:'+str(statistics.median(success_query))+'\n\n')

        f.write('query\n')
        for cnt in range(1,img_size+1):
            f.write('{0:<6}  {1:<6}'.format(cnt*cnt*img_chan,q[cnt])+'\n')
        f.write('failed  '+str(q[0])+'\n')

    return X_adv



##
# Preparing model and data
##
print('\nPreparing CIFAR-10 data')

(X_train, y_train), (X_test, y_test), (X_valid, y_valid) = load_cifar10()

print('\nConstruction graph')

env = model_cifar10()

print('\nInitializing graph')

sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())
sess.run(tf.local_variables_initializer())

print('\nLoading saved model')

env.saver.restore(sess, 'models/model_cifar10/{}'.format('model_cifar10'))

print('\nEvaluating on clean data')

evaluate(sess, env, X_test, y_test)



##
# Attack
##
print('\nExcluding misclassification samples')

# 1000 samples -> 0:1234
(X_test, y_test) = exclude_miss(sess, env, X_test, y_test, 0, 100)
evaluate(sess, env, X_test, y_test)

print('\nGenerating adversarial data')

X_adv = make_adv(sess, env, X_test, y_test)

print('\nEvaluating on adversarial data')

evaluate(sess, env, X_adv, y_test)



##
# Output adv images
##
print('\nOutput')

sample_img(sess, env, X_adv, X_test, y_test, name='cifar10')
out_(sess, env, X_adv, X_test, y_test, name='cifar10', attack='YY')