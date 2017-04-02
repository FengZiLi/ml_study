import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

#老实说，目前还没搞懂这是在干啥呢
def addLayer(inputs,inputSize,outputsSize,activeFunction=None):
    Weights=tf.Variable(tf.random_normal([inputSize,outputsSize]))
    biases=tf.Variable(tf.zeros([1,outputsSize])+0.1)
    Wxplusb=tf.matmul(inputs,Weights)+biases # 让输入乘以权重+偏差
    if activeFunction is not None:
        return activeFunction(Wxplusb)  #应用激励函数
    else:
        return Wxplusb


# 创建训练数据
x_data=np.linspace(-1,1,300)[:,np.newaxis]
nosies=np.random.normal(0,0.1,x_data.shape)
y_data=np.square(x_data)-0.5+nosies
#训练数据创建完毕

#定义神经网络的placeholder
xs=tf.placeholder(tf.float32,[None,1])
ys=tf.placeholder(tf.float32,[None,1])

#定义隐藏层
layer1=addLayer(xs,1,10,tf.nn.relu)
prediction=addLayer(layer1,10,1)

loss=tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction),reduction_indices=[1]))# 这个reduction_indices是干啥的
train=tf.train.GradientDescentOptimizer(0.1).minimize(loss)

#初始化变量
init=tf.global_variables_initializer()

#绘图
figure=plt.figure()
ax=figure.add_subplot(1,1,1)
ax.scatter(x_data,y_data)
plt.ion()
plt.show()


with tf.Session() as session:
    session.run(init)
    for _ in range(2000):
        session.run(train,feed_dict={xs:x_data,ys:y_data})
        if _ % 50 ==0:
            print(session.run(loss,feed_dict={xs:x_data,ys:y_data}))
            try:
                ax.lines.remove(lines[0])
            except:
                pass
            lines=ax.plot(x_data,session.run(prediction,feed_dict={xs:x_data,ys:y_data}),'r-',lw=5)
            plt.pause(0.3)

    plt.pause(5)
