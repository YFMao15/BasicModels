import os
import sys
import tensorflow as tf
import tensorboard as tb
import numpy as np

# odel of MLP
def multi_layer_perceptron(X,layer_dims):  
    # full connected for all variables
    # parameter settings in layers.dense(activation=) is non-linear transformation
    if isinstance(layer_dims,tuple): 
        fc1=tf.layers.dense(
        inputs=X,
        units=layer_dims[0],
        activation=tf.nn.sigmoid,
        kernel_initializer=None,
        bias_initializer=tf.zeros_initializer(),
        trainable=True,
        name='fc1')

        fc2=tf.layers.dense(
        inputs=fc1,
        units=layer_dims[1],
        activation=None,
        kernel_initializer=None,
        bias_initializer=tf.zeros_initializer(),
        trainable=True,
        name='fc2')
    
    # the number of output dimension should be number of classes
    # the meaning is the probability of each class
        Y_eval=tf.layers.dense(
        inputs=fc2,
        units=layer_dims[2],
        activation=tf.nn.sigmoid,
        kernel_initializer=None,
        bias_initializer=tf.zeros_initializer(),
        trainable=True,
        name='out')

    elif isinstance(layer_dims,int)==1:
        Y_eval=tf.layers.dense(
        inputs=X,
        units=layer_dims,
        activation=tf.nn.sigmoid,
        kernel_initializer=None,
        bias_initializer=tf.zeros_initializer(),
        trainable=True,
        name='fc1')
    return Y_eval

# internal functions for neural network
# if MSE is the loss function, than the label should be hot vector instead of an integer
def loss_function(mode,Y,Y_eval):
    if mode=='squared_error':
        loss=tf.reduce_sum(tf.square(Y-Y_eval))
    elif mode=='cross_entropy':
        cross_entropy=tf.nn.softmax_cross_entropy_with_logits(
            logits=Y_eval,labels=Y,name='xentropy')
        loss=tf.reduce_mean(cross_entropy,name='loss')
    return loss

def training(mode,loss,learning_rate):
    if mode=='gradient_descend':
        optimizer=tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
    elif mode=='adam_optimize':    
        optimizer=tf.train.AdamOptimizer(learning_rate=learning_rate)
    global_step=tf.Variable(initial_value=0,trainable=False,name='global_step')
    train_optimizer=optimizer.minimize(loss=loss,global_step=global_step)    
    return train_optimizer

def evaluation(Y,Y_eval):
    _,Y_class=tf.nn.top_k(input=Y,k=1,sorted=False,name='result_class')
    _,Y_eval_class=tf.nn.top_k(input=Y_eval,k=1,sorted=False,name='predicted_class')
    difference=tf.to_double(tf.count_nonzero(tf.subtract(Y_class,Y_eval_class)))
    accuracy=(tf.to_double(float(Y_class.shape[0].value))-difference)/tf.to_double(float(Y_class.shape[0].value))
    returns={}
    returns['Y_eval']=Y_eval
    returns['Y_class']=Y_class
    returns['Y_eval_class']=Y_eval_class
    returns['accuracy']=accuracy
    return returns

# generate batch for the MLP model
def get_batch(data,labels,batch_size):
    data=tf.cast(data,tf.float64)
    labels=tf.cast(labels,tf.float64)
    input_queue=tf.train.slice_input_producer([data,labels],shuffle=True)
    input_data_batch,input_label_batch=tf.train.batch(
        tensors=input_queue,
        batch_size=batch_size,
        num_threads=16)
    return input_data_batch,input_label_batch


def neuron_model(mode,*args,**kwargs):
    if mode=='train_test':
        np_train_data=args[0]
        np_test_data=args[1]
        max_step=kwargs['max_step']
        layer_dims=kwargs['layer_dims']
        learning_rate=kwargs['learning_rate']
        batch_size=kwargs['batch_size']

        keys=list(np_train_data.keys())
        keys_num=len(keys)
        input_dim=np_train_data[keys[0]].shape[0]
        for key_num in range(keys_num):
            train_num=np_train_data[keys[key_num]].shape[1]
            test_num=np_test_data[keys[key_num]].shape[1]
            if key_num==0:
                train_labels=np.zeros((train_num,keys_num))
                test_labels=np.zeros((test_num,keys_num))
                train_data=np_train_data[keys[key_num]].T
                test_data=np_test_data[keys[key_num]].T
                train_labels[:,key_num]=1.0
                test_labels[:,key_num]=1.0
            elif key_num>0:
                train_data=np.vstack((train_data,np_train_data[keys[key_num]].T))
                test_data=np.vstack((test_data,np_test_data[keys[key_num]].T))
                temp_train_labels=np.zeros((train_num,keys_num))
                temp_test_labels=np.zeros((test_num,keys_num))
                temp_train_labels[:,key_num]=1.0
                temp_test_labels[:,key_num]=1.0
                train_labels=np.vstack((train_labels,temp_train_labels))
                test_labels=np.vstack((test_labels,temp_test_labels))

        # get batch from the datasets
        train_data_batch,train_label_batch=get_batch(train_data,train_labels,batch_size)
        test_data_batch,test_label_batch=get_batch(test_data,test_labels,batch_size)

        # set the placeholders
        X=tf.placeholder(tf.float64,shape=[batch_size,input_dim])
        Y=tf.placeholder(tf.float64,shape=[batch_size,keys_num])

        Y_eval=multi_layer_perceptron(X,layer_dims)
        loss=loss_function('squared_error',Y,Y_eval)  
        #loss=loss_function('cross_entropy',Y,Y_eval)  
        accuracy=evaluation(Y,Y_eval)
        train_optimizer=training('gradient_descend',loss,learning_rate)
        #train_optimizer=training('adam_optimize',loss,learning_rate)

        with tf.Session() as sess:
            # prepare the writers 
            print('Initializing variables for kernel and bias.')
            saver=tf.train.Saver()
            sess.run(tf.global_variables_initializer())
            # thread and coordinator cannot be ignored
            coord=tf.train.Coordinator()
            threads=tf.train.start_queue_runners(sess=sess,coord=coord)
            directory=os.path.dirname(__file__)
            train_writer=tf.summary.FileWriter(directory,sess.graph)
            test_writer=tf.summary.FileWriter(directory,sess.graph)
            summary=tf.summary.merge_all()
            print('Training process starts.\n')
            for curr_step in range(max_step):
                train_samples,train_labels=sess.run([train_data_batch,train_label_batch])
                _,train_loss,train_returns=sess.run([train_optimizer,loss,accuracy], 
                    feed_dict={X:train_samples,Y:train_labels})
                if curr_step % 200 ==0:
                    print('Current step is %d' % (curr_step+1))
                    test_samples,test_labels=sess.run([test_data_batch,test_label_batch])
                    test_loss,test_returns=sess.run([loss,accuracy], 
                        feed_dict={X:test_samples,Y:test_labels})
                    print('Training set: Loss--%.3f, Accuracy--%.3f' % (train_loss,train_returns['accuracy']))
                    print('Testing set: Loss--%.3f, Accuracy--%.3f' % (test_loss,test_returns['accuracy']))
                    train_summary=sess.run(summary,feed_dict={X:train_samples,Y:train_labels})    
                    train_writer.add_summary(train_summary,curr_step+1)
                    test_summary=sess.run(summary,feed_dict={X:test_samples,Y:test_labels})
                    test_writer.add_summary(test_summary,curr_step+1)
                    model_name=mode+str(input_dim)+'model.ckpt'
                    checkpoint_path=os.path.join(directory,model_name)
                    saver.save(sess,checkpoint_path,global_step=curr_step)
            coord.request_stop()           
            coord.join(threads)
            sess.close()
        tf.reset_default_graph()
                
