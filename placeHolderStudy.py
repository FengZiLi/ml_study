import tensorflow as tf

input1=tf.placeholder(tf.float32) #定义占位符变量
input2=tf.placeholder(tf.float32)

output=tf.add(input1,input2)

with tf.Session() as session:
    result=session.run(output,feed_dict={input1:[input('输入第一个参数')],input2:[input('输入第二个参数')]})
    print(result)

