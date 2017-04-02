'''
本节练习周莫烦的预测y=x*0.1+0.3
'''
import tensorflow as tf
import  numpy as np


#构造数据
x_data=np.random.rand(2000).astype(np.float32)
y_data= x_data * 0.1 + 0.3
#样本数据构造完毕

#创建神经网络结构
Weights=tf.Variable(tf.random_uniform([1],-100,100))  #给一个计算范围，让机器随机初始化以后逐渐靠近，这个区间随意，机器最终会训练得到正常的值
biases=tf.Variable(tf.zeros([1])) # 默认成0,但是也要做成一个变量——这让机器后面去修正

y=Weights * x_data + biases

loss = tf.reduce_mean(tf.square(y-y_data)) # 最终是要优化这个差值


optimizer=tf.train.GradientDescentOptimizer(0.5) # 优化器，这里的值是干啥的？说是0-1之间:数值越大，得到最终结果的计算次数越少；1时算不出
train=optimizer.minimize(loss) #初始化训练用的优化器，优化器用来降低误差

init=tf.global_variables_initializer() #设置初始化语句

# 结构创建完毕

with tf.Session() as session:
    session.run(init) #这里真正的运行初始化，开始初始化变量
    for i in range(2000):
        session.run(train) #执行训练器
        print(i,'  y=',session.run(Weights),'*x+',session.run(biases))