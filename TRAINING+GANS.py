
# coding: utf-8

# The idea of this code is to train a 'smarter' GAN. While most GAN train both the discriminator and the generator at the same time-- both of which start out dumb-- our approach is to train the generator first with a binary discriminator and then replace the discriminator with a smarter(pretrained) one that does localization. This will require replacing the min max function. 

# In[45]:

import tensorflow as tf


# In[46]:

#create something called filepaths_new that has the file paths of all the images 
#just need to replace the directory to run 

directory = '/Users/caitlinshener/Desktop/RIPS/tire_detection/test_images/img'
import os, fnmatch
filepaths_new = []
listOfFiles = os.listdir(directory)  
pattern = "*.jpg"  
for entry in listOfFiles:  
    if fnmatch.fnmatch(entry, pattern):
        #relDir = os.path.relpath(dir_, directory)
        #relFile = os.path.join(relDir, entry)
        filepaths_new.append(directory + "/" + entry)


# In[47]:

tf.reset_default_graph()
# lets assume that we have the same batch size 
# TO DO: is this is batch size we want? 
batch_size = 64
n_noise = 64


# In[48]:

#changed the X_in to be the size of our images 
X_in = tf.placeholder(dtype=tf.float32, shape=[None, 150, 150], name='X')
noise = tf.placeholder(dtype=tf.float32, shape=[None, n_noise])
#The keep_prob variable will be used by our dropout layers, 
#which we introduce for more stable learning outcomes.
keep_prob = tf.placeholder(dtype=tf.float32, name='keep_prob')
is_training = tf.placeholder(dtype=tf.bool, name='is_training')


# In[49]:

#leaky ReLu
def lrelu(x):
    return tf.maximum(x, tf.multiply(x, 0.2))


# In[50]:

#used to compute losses 
def binary_cross_entropy(x, z):
    eps = 1e-12
    return (-(x * tf.log(z + eps) + (1. - x) * tf.log(1. - z + eps)))


# In[51]:

#The discriminator

def discriminator(img_in, reuse=None, keep_prob=keep_prob):
    activation = lrelu
    with tf.variable_scope("discriminator", reuse=reuse):
        #changed to 150x150 image
        x = tf.reshape(img_in, shape=[-1, 150, 150, 3])
        #TO DO-- should we change the number of filters? 
        x = tf.layers.conv2d(x, kernel_size=5, filters=256, strides=2, padding='same', activation=activation)
        x = tf.layers.dropout(x, keep_prob)
        x = tf.layers.conv2d(x, kernel_size=5, filters=128, strides=1, padding='same', activation=activation)
        x = tf.layers.dropout(x, keep_prob)
        x = tf.layers.conv2d(x, kernel_size=5, filters=64, strides=1, padding='same', activation=activation)
        x = tf.layers.dropout(x, keep_prob)
        x = tf.contrib.layers.flatten(x)
        x = tf.layers.dense(x, units=128, activation=activation)
        x = tf.layers.dense(x, units=1, activation=tf.nn.sigmoid)
        return x
    


# In[52]:

#The generator 

def generator(z, keep_prob=keep_prob, is_training=is_training):
    activation = lrelu
    momentum = 0.9
    with tf.variable_scope("generator", reuse=None):
        x = z
        
        d1 = 4#3
        d2 = 3
        
        x = tf.layers.dense(x, units=d1 * d1 * d2, activation=activation)
        x = tf.layers.dropout(x, keep_prob)      
        x = tf.contrib.layers.batch_norm(x, is_training=is_training, decay=momentum)  
        
        x = tf.reshape(x, shape=[-1, d1, d1, d2])
        x = tf.image.resize_images(x, size=[10, 10])
        
        #it looks like the filters are the same as in the discriminator 
        #if we want to change the filters we should also change above
        
        x = tf.layers.conv2d_transpose(x, kernel_size=5, filters=256, strides=2, padding='same', activation=activation)
        x = tf.layers.dropout(x, keep_prob)
        x = tf.contrib.layers.batch_norm(x, is_training=is_training, decay=momentum)
        x = tf.layers.conv2d_transpose(x, kernel_size=5, filters=128, strides=2, padding='same', activation=activation)
        x = tf.layers.dropout(x, keep_prob)
        x = tf.contrib.layers.batch_norm(x, is_training=is_training, decay=momentum)
        x = tf.layers.conv2d_transpose(x, kernel_size=5, filters=64, strides=1, padding='same', activation=activation)
        x = tf.layers.dropout(x, keep_prob)
        x = tf.contrib.layers.batch_norm(x, is_training=is_training, decay=momentum)
        x = tf.layers.conv2d_transpose(x, kernel_size=5, filters=3, strides=1, padding='same', activation=tf.nn.sigmoid)
        return x


# In[53]:

def next_batch(num=64, data=filepaths_new):
    idx = np.arange(0 , len(data))
    np.random.shuffle(idx)
    idx = idx[:num]
    data_shuffle = [imread(data[i]) for i in idx]

    shuffled = np.asarray(data_shuffle)
    
    return np.asarray(data_shuffle)
        


# In[54]:

g = generator(noise, keep_prob, is_training)
print(g)
d_real = discriminator(X_in)
d_fake = discriminator(g, reuse=True)

vars_g = [var for var in tf.trainable_variables() if var.name.startswith("generator")]
vars_d = [var for var in tf.trainable_variables() if var.name.startswith("discriminator")]


d_reg = tf.contrib.layers.apply_regularization(tf.contrib.layers.l2_regularizer(1e-6), vars_d)
g_reg = tf.contrib.layers.apply_regularization(tf.contrib.layers.l2_regularizer(1e-6), vars_g)

loss_d_real = binary_cross_entropy(tf.ones_like(d_real), d_real)
loss_d_fake = binary_cross_entropy(tf.zeros_like(d_fake), d_fake)
loss_g = tf.reduce_mean(binary_cross_entropy(tf.ones_like(d_fake), d_fake))

loss_d = tf.reduce_mean(0.5 * (loss_d_real + loss_d_fake))

update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(update_ops):
    optimizer_d = tf.train.RMSPropOptimizer(learning_rate=0.0001).minimize(loss_d + d_reg, var_list=vars_d)
    optimizer_g = tf.train.RMSPropOptimizer(learning_rate=0.0002).minimize(loss_g + g_reg, var_list=vars_g)
sess = tf.Session()
sess.run(tf.global_variables_initializer())


# In[ ]:

for i in range(60000):
    train_d = True
    train_g = True
    keep_prob_train = 0.6 # 0.5
    
    
    n = np.random.uniform(0.0, 1.0, [batch_size, n_noise]).astype(np.float32)   
    batch = [b for b in next_batch(num=batch_size)]  
    
    d_real_ls, d_fake_ls, g_ls, d_ls = sess.run([loss_d_real, loss_d_fake, loss_g, loss_d], feed_dict={X_in: batch, noise: n, keep_prob: keep_prob_train, is_training:True})
    
    d_fake_ls_init = d_fake_ls
    
    d_real_ls = np.mean(d_real_ls)
    d_fake_ls = np.mean(d_fake_ls)
    g_ls = g_ls
    d_ls = d_ls
        
    if g_ls * 1.35 < d_ls:
        train_g = False
        pass
    if d_ls * 1.35 < g_ls:
        train_d = False
        pass
    
    if train_d:
        sess.run(optimizer_d, feed_dict={noise: n, X_in: batch, keep_prob: keep_prob_train, is_training:True})
        
        
    if train_g:
        sess.run(optimizer_g, feed_dict={noise: n, keep_prob: keep_prob_train, is_training:True})
        
        
    if not i % 10:
        print (i, d_ls, g_ls)
        if not train_g:
            print("not training generator")
        if not train_d:
            print("not training discriminator")
        gen_imgs = sess.run(g, feed_dict = {noise: n, keep_prob: 1.0, is_training:False})
        imgs = [img[:,:,:] for img in gen_imgs]
        #montonge is on git but we want to save the images instead 
        #m = montage(imgs)
        #m = imgs[0]
        #plt.axis('off')
        #plt.imshow(m, cmap='gray')
        #plt.show()


# In[56]:

#assuming imgs is a list of images we want... save these so we can look at them 
#not able to test this because haven't generated images
new_path2 = 'generated_img/'
if not os.path.exists(new_path2):
   os.makedirs(new_path2)
for im in imgs:
   if not os.path.isdir(im):
       im.save(new_path2 + im,"png")


# In[ ]:



