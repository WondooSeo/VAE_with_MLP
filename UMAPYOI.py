import tensorflow as tf
import numpy as np
import h5py
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
import os
import shutil

""""""
## Real Data ##
""""""
gpu_number = "3"
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = gpu_number
code_filename = 'UMAPYOI.py'

""""""
## Setting Parameter ##
""""""
learning_rate = 0.0001       # ETA
training_steps = 100001      # Step
batch_size = 64              # Bathc Size
# beta1 = 0.5
# beta2 = 0.999
# ------------------- #
n_train_fVh = 2
nnn = 8
test_size = nnn**2
test_size2 = 30
# n_delta=12
# delta=1
n_delta = 8
delta = 1.5
# ------------------- #
training_style = 2
# ------------------- #
input_channel = 1
img_size_factor = 2
input_sx = 64*img_size_factor
input_sy = 64*img_size_factor
# ------------------- #
img_constant = 255
# ------------------- #
grid_num_2d = 5*2+1          # Why 11?

""""""
## Experiment setting ##
""""""
training_name = 'deep_EIT_' + str(input_sx)
exp_date = '181011'
exp_num = '2'
exp_dir = './experiment_' + exp_date + '_' + exp_num + '_' + training_name
log_dir = './logs'

""""""
# Pre-Trained Weights Setting
""""""
use_pre_trained = 'off'
weights_date = '180712'
weights_num = '2'
steps_num = '20400'
weights_file_name = "./experiment_" + weights_date + '_' + weights_num + '_' + training_name + '/' + training_name + "_training_weight_" + weights_date + '_' + weights_num + ".ckpt" + "-" + steps_num

if img_size_factor == 1:
    w_sx = 5
    # filename_train='./[0]data/save/EIT_manifold_data_64_180711'+'.mat'
else:
    w_sx = 11
    filename_train = './[0]data/save/EIT_deep_data_128_real'+'.mat'

""""""
# Training Data Setting
""""""
arrays1 = {}
f1 = h5py.File(filename_train, 'r')
for k1, v1 in f1.items():
    arrays1[k1] = np.array(v1)
    print(v1)

""""""
# Modify Array Made in MATLAB
""""""
train_X = arrays1['train_X_input']
train_X = np.reshape(train_X, [-1, input_sx, input_sy])
train_X = np.transpose(train_X, [0, 2, 1])
train_X = np.reshape(train_X, [-1, input_channel*input_sx*input_sy])
print('train_X', train_X.shape)
train_X = train_X/img_constant

test_X = arrays1['test_X_input']
test_X = np.reshape(test_X, [-1, input_sx, input_sy])
test_X = np.transpose(test_X, [0, 2, 1])
test_X = np.reshape(test_X, [-1, input_channel*input_sx*input_sy])
print('test_X', test_X.shape)
test_X = test_X/img_constant

train_V = arrays1['train_V_input']
print('train_V', train_V.shape)
test_V = arrays1['test_V_input']
print('test_V', test_V.shape)

num_data=train_X.shape[0]
total_batch = int(num_data / batch_size)

#Selected Test data

###### new_part
test_num_start=1
test_num_start=test_num_start-1
feed_X=test_X[test_num_start:test_num_start+test_size2]
print('feed_X', feed_X.shape)
feed_X=feed_X

feed_V=test_V[test_num_start:test_num_start+test_size2]
print('feed_V', feed_V.shape)


def init_weights(size,name):
    return tf.Variable(tf.random_normal(size, stddev=0.01),name=name)

def batch_norm(summed_input, is_training):
    return tf.layers.batch_normalization(summed_input, training=is_training)

cdict = {'red':   ((0.0,  1.0, 1.0),
                   (0.2,  1.0, 1.0),
                   (0.5,  0.0, 0.0),
                   (0.8,  0.0, 0.0),
                   (1.0,  1.0, 1.0)),

         'green': ((0.0,  1.0, 1.0),
                   (0.2, 0.0, 0.0),
                   (0.5, 0.0, 0.0),
                   (0.8, 0.0, 0.0),
                   (1.0,  1.0, 1.0)),

         'blue':  ((0.0,  1.0, 1.0),
                   (0.2, 0.0, 0.0),
                   (0.5, 0.0, 0.0),
                   (0.8, 1.0, 1.0),
                   (1.0,  1.0, 1.0))}

custom_color_map = LinearSegmentedColormap('custom_color_map', cdict)

#plot setting
cmap=custom_color_map
vmax=1
vmin=-1

#Plot parameter
fig_space=0.05
single_size=1
def plot(samples,c_map,v_max,v_min,size_x,size_y):
    fig = plt.figure(figsize=(single_size*(size_y+fig_space), single_size*(size_x+fig_space)))
    gs = gridspec.GridSpec(size_x, size_y)
    gs.update(wspace=fig_space, hspace=fig_space)
    for i, sample in enumerate(samples):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        if input_channel==1:
            sample=sample.reshape(input_sy,input_sx)
        else:
            sample=sample.reshape(input_sy,input_sx,input_channel)
        plt.imshow(sample, cmap=c_map, vmax=v_max, vmin=v_min)
    return fig

depth_param=8

h_dim11=4*depth_param
h_dim12=8*depth_param
h_dim13=16*depth_param
h_dim14=32*depth_param
z_dim=2
h_dim20=32*depth_param
h_dim21=16*depth_param
h_dim22=8*depth_param
h_dim23=4*depth_param

b_sx=4*img_size_factor #size of bottom feature map

#weight setting parameter
w11=init_weights([w_sx, w_sx, input_channel, h_dim11],'w11')
w12=init_weights([w_sx, w_sx, h_dim11, h_dim12],'w12')
w13=init_weights([w_sx, w_sx, h_dim12, h_dim13],'w13')
w14=init_weights([w_sx, w_sx, h_dim13, h_dim14],'w14')
w_mu=init_weights([b_sx,b_sx,h_dim14, z_dim],'w_mu')
w_var=init_weights([b_sx,b_sx,h_dim14, z_dim],'w_var')

w20=init_weights([b_sx, b_sx, h_dim20, z_dim],'w20')
w21=init_weights([w_sx, w_sx, h_dim21, h_dim20],'w21')
w22=init_weights([w_sx, w_sx, h_dim22, h_dim21],'w22')
w23=init_weights([w_sx, w_sx, h_dim23, h_dim22],'w23')
w24=init_weights([w_sx, w_sx, input_channel, h_dim23],'w24')

theta_VAE=[w11,w12,w13,w14,w_mu,w_var,w20,w21,w22,w23,w24]
theta_VAE_saver={'w11': w11, 'w12': w12, 'w13': w13,'w14': w14, 'w_mu': w_mu , 'w_var': w_var,
                    'w20': w20, 'w21': w21, 'w22': w22,'w23': w23, 'w24': w24}

# parameter of fVh

v_dim=208
fff=2
h_dim_t1=128*fff
h_dim_t2=64*fff
h_dim_t3=32*fff
h_dim_t4=16*fff

w_t1=init_weights([v_dim, h_dim_t1],'w_t1')
w_t2=init_weights([h_dim_t1, h_dim_t2],'w_t2')
w_t3=init_weights([h_dim_t2, h_dim_t3],'w_t3')
w_t4=init_weights([h_dim_t3, h_dim_t4],'w_t4')
w_t_mu=init_weights([h_dim_t4, z_dim],'w_t_mu')

theta_fVh=[w_t1,w_t2,w_t3,w_t4,w_t_mu]
theta_fVh_saver={'w_t1': w_t1, 'w_t2': w_t2, 'w_t3': w_t3, 'w_t4': w_t4, 'w_t_mu': w_t_mu}

theta_feit=[w11,w12,w13,w_mu,w_var,w20,w21,w22,w23,w_t1,w_t2,w_t3,w_t4,w_t_mu]
theta_feit_saver={'w11': w11, 'w12': w12, 'w13': w13,'w14': w14, 'w_mu': w_mu , 'w_var': w_var, 'w20': w20, 'w21': w21, 'w22': w22,'w23': w23, 'w24': w24,
        'w_t1': w_t1, 'w_t2': w_t2, 'w_t3': w_t3, 'w_t4': w_t4, 'w_t_mu': w_t_mu}


# placeholder
X = tf.placeholder(dtype= tf.float32, shape = (None, input_sx*input_sy*input_channel))
Z = tf.placeholder(dtype= tf.float32, shape = (None, z_dim))
V = tf.placeholder(dtype= tf.float32, shape = (None, v_dim))
isTraining = tf.placeholder(dtype= tf.bool)

def encoder_VAE(X,w11,w12,w13,w14,w_mu,w_var,isTraining):
    print('X', X.shape)
    X=tf.reshape(X,[-1, input_sx, input_sy, input_channel])
    print('X', X.shape)

    l11_c=tf.nn.conv2d(X,w11,strides=[1,2,2,1],padding='SAME')
    l11_c=tf.nn.relu(batch_norm(l11_c, isTraining))
    print('l11_c', l11_c.shape)

    l21_c=tf.nn.conv2d(l11_c,w12,strides=[1,2,2,1],padding='SAME')
    l21_c=tf.nn.relu(batch_norm(l21_c, isTraining))
    print('l21_c', l21_c.shape)

    l31_c=tf.nn.conv2d(l21_c,w13,strides=[1,2,2,1],padding='SAME')
    l31_c=tf.nn.relu(batch_norm(l31_c, isTraining))
    print('l31_c', l31_c.shape)

    l41_c=tf.nn.conv2d(l31_c,w14,strides=[1,2,2,1],padding='SAME')
    l41_c=tf.nn.relu(batch_norm(l41_c, isTraining))
    print('l41_c', l41_c.shape)

    z_mu=tf.nn.conv2d(l41_c,w_mu,strides=[1,1,1,1],padding='VALID')
    z_mu=batch_norm(z_mu, isTraining)
    print('z_mu', z_mu.shape)
    z_mu=tf.reshape(z_mu,[-1,z_dim])
    print('z_mu', z_mu.shape)

    z_log_var=tf.nn.conv2d(l41_c,w_var,strides=[1,1,1,1],padding='VALID')
    z_log_var=batch_norm(z_log_var, isTraining)
    print('z_log_var', z_log_var.shape)
    z_log_var=tf.reshape(z_log_var,[-1,z_dim])
    print('z_log_var', z_log_var.shape)

    return z_mu, z_log_var

def sample_z(mu, log_var):
    z_randn = tf.random_normal(shape=tf.shape(mu))
    print('z_randn', z_randn.shape)
    z_sample = mu + tf.exp(log_var/2) * z_randn
    print('z_sample', z_sample.shape)

    return z_sample

def decoder_VAE(Z,w20,w21,w22,w23,w24,batch_size_deconv,isTraining):
    print('Z', Z.shape)
    Z=tf.reshape(Z,[-1, 1,1,z_dim])
    print('Z', Z.shape)

    l5_0 = tf.nn.conv2d_transpose(Z, w20, output_shape = (batch_size_deconv, b_sx*1, b_sx*1, h_dim20), strides = [1, 1, 1, 1], padding = 'VALID')
    l5_0 = tf.nn.relu(batch_norm(l5_0, False))
    print('l5_0', l5_0.shape)

    l51_dc = tf.nn.conv2d_transpose(l5_0, w21, output_shape = (batch_size_deconv, b_sx*2, b_sx*2, h_dim21), strides = [1, 2, 2, 1], padding = 'SAME')
    l51_dc = tf.nn.relu(batch_norm(l51_dc, False))
    print('l51_dc', l51_dc.shape)

    l61_dc = tf.nn.conv2d_transpose(l51_dc, w22, output_shape = (batch_size_deconv, b_sx*4, b_sx*4, h_dim22), strides = [1, 2, 2, 1], padding = 'SAME')
    l61_dc = tf.nn.relu(batch_norm(l61_dc, False))
    print('l61_dc', l61_dc.shape)

    l71_dc = tf.nn.conv2d_transpose(l61_dc, w23, output_shape = (batch_size_deconv, b_sx*8, b_sx*8, h_dim23), strides = [1, 2, 2, 1], padding = 'SAME')
    l71_dc = tf.nn.relu(batch_norm(l71_dc, False))
    print('l71_dc', l71_dc.shape)

    l81_dc = tf.nn.conv2d_transpose(l71_dc, w24, output_shape = (batch_size_deconv, b_sx*16, b_sx*16, input_channel), strides = [1, 2, 2, 1], padding = 'SAME')
    l81_dc = batch_norm(l81_dc, False)

    print('l81_dc', l81_dc.shape)
    logits = tf.reshape(l81_dc,[-1, input_sx*input_sy*input_channel])
    print('logits', logits.shape)
    prob = tf.nn.tanh(logits)
    print('prob', prob.shape)

    return logits, prob

def encoder_fVh(V,w_t1,w_t2,w_t3,w_t4,w_t_mu,isTraining):
    print('V',V.shape)

    l_t1=tf.matmul(V,w_t1)
    l_t1 = tf.nn.relu(batch_norm(l_t1, isTraining))
    print('l_t1',l_t1.shape)

    l_t2=tf.matmul(l_t1,w_t2)
    l_t2 = tf.nn.relu(batch_norm(l_t2, isTraining))
    print('l_t2',l_t2.shape)

    l_t3=tf.matmul(l_t2,w_t3)
    l_t3 = tf.nn.relu(batch_norm(l_t3, isTraining))
    print('l_t3',l_t3.shape)

    l_t4=tf.matmul(l_t3,w_t4)
    l_t4 = tf.nn.relu(batch_norm(l_t4, isTraining))
    print('l_t4',l_t4.shape)

    z_t_mu=tf.matmul(l_t4,w_t_mu)
    z_t_mu = batch_norm(z_t_mu, isTraining)
    print('z_t_mu', z_t_mu.shape)

    return z_t_mu

# VAE function
z_mu, z_log_var = encoder_VAE(X,w11,w12,w13,w14,w_mu,w_var,isTraining)
latent_VAE = sample_z(z_mu,z_log_var)
logits_X_VAE, inference_X_VAE = decoder_VAE(latent_VAE,w20,w21,w22,w23,w24,batch_size,isTraining)

# fVh function
latent_fVh = encoder_fVh(V,w_t1,w_t2,w_t3,w_t4,w_t_mu,isTraining)
_, inference_V_VAE = decoder_VAE(latent_fVh,w20,w21,w22,w23,w24,batch_size,isTraining)
_, inference_V_VAE_test = decoder_VAE(latent_fVh,w20,w21,w22,w23,w24,test_size,isTraining)

#inference
_, inference_X_VAE_test = decoder_VAE(latent_VAE,w20,w21,w22,w23,w24,test_size,isTraining)
_, inference_Z_VAE_test = decoder_VAE(Z,w20,w21,w22,w23,w24,test_size,isTraining)
_, inference_X_VAE_test2 = decoder_VAE(latent_VAE,w20,w21,w22,w23,w24,test_size2,isTraining)
_, inference_Z_VAE_test2 = decoder_VAE(Z,w20,w21,w22,w23,w24,test_size2,isTraining)
_, inference_V_VAE_test2 = decoder_VAE(latent_fVh,w20,w21,w22,w23,w24,test_size2,isTraining)
_, inference_Z_VAE_test3 = decoder_VAE(Z,w20,w21,w22,w23,w24,z_dim*(2*n_delta+1),isTraining)
_, inference_Z_VAE_test_2d = decoder_VAE(Z,w20,w21,w22,w23,w24,grid_num_2d**2,isTraining)

# Loss
loss_KL = 0.5 * tf.reduce_sum(1.0 + z_log_var - tf.square(z_mu) - tf.exp(z_log_var), 1)
loss_recon = tf.nn.l2_loss(inference_X_VAE - X)
loss_VAE = tf.reduce_mean(-loss_KL+loss_recon)

loss_fVh_recon = tf.nn.l2_loss(X - inference_V_VAE)
loss_fVh_latent = tf.nn.l2_loss(z_mu - latent_fVh)

loss_fVh = tf.reduce_mean(loss_fVh_recon + loss_fVh_latent)

loss_feit = 0.5*loss_VAE + loss_fVh

# Training Part

optimizer_VAE = tf.train.AdamOptimizer(learning_rate)
optimizer_fVh = tf.train.AdamOptimizer(learning_rate)
optimizer_feit = tf.train.AdamOptimizer(learning_rate)

update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(update_ops):
    if training_style==1:
        train_op_VAE = optimizer_VAE.minimize(loss_VAE, var_list=theta_VAE)
        train_op_fVh = optimizer_fVh.minimize(loss_fVh, var_list=theta_fVh)
    elif training_style==2:
        train_op_feit=optimizer_feit.minimize(loss_feit,var_list=theta_feit)

# Tensorboard Setting
if training_style==1:
    loss_steps_VAE = tf.placeholder("float")
    loss_steps_fVh = tf.placeholder("float")
    summary_loss_op_VAE = tf.summary.merge([tf.summary.scalar("loss_VAE",loss_steps_VAE)])
    summary_loss_op_fVh = tf.summary.merge([tf.summary.scalar("loss_fVh",loss_steps_fVh)])
elif training_style==2:
    loss_steps_feit = tf.placeholder("float")
    summary_loss_op_feit = tf.summary.merge([tf.summary.scalar("loss_feit",loss_steps_feit)])




saver = tf.train.Saver()
with tf.Session() as sess:

    sess.run(tf.global_variables_initializer())
    if use_pre_trained=='on':
        saver.restore(sess,weights_file_name)
    writer = tf.summary.FileWriter(exp_dir+log_dir)
    shutil.copyfile(code_filename, exp_dir + "./"+ code_filename)

    if not os.path.exists(exp_dir+'./out1/'):
        os.makedirs(exp_dir+'./out1/')
    if not os.path.exists(exp_dir+'./out2/'):
        os.makedirs(exp_dir+'./out2/')
    if not os.path.exists(exp_dir+'./test_recon/'):
        os.makedirs(exp_dir+'./test_recon/')
    if not os.path.exists(exp_dir+'./out3/'):
        os.makedirs(exp_dir+'./out3/')

    for steps in range(training_steps):
        if training_style==1:
            #Train VAE and fVh alternatively
            rand_idx_VAE=np.random.randint(num_data,size=batch_size)
            X_batch_VAE = train_X[rand_idx_VAE]
            V_batch_VAE = train_V[rand_idx_VAE]
            feed_dict_VAE={X: X_batch_VAE, V: V_batch_VAE, isTraining: True}
            _, VAE_loss_val = sess.run([train_op_VAE, loss_VAE],feed_dict=feed_dict_VAE)

            for i in range(n_train_fVh):
                rand_idx_fVh=np.random.randint(num_data,size=batch_size)
                X_batch_fVh = train_X[rand_idx_fVh]
                V_batch_fVh = train_V[rand_idx_fVh]
                feed_dict_fVh={X: X_batch_fVh, V: V_batch_fVh, isTraining: True}
                _, fVh_loss_val = sess.run([train_op_fVh, loss_fVh],feed_dict=feed_dict_fVh)
        elif training_style==2:
            #Train VAE and fVh simultaneously
            rand_idx_feit=np.random.randint(num_data,size=batch_size)
            X_batch_feit = train_X[rand_idx_feit]
            V_batch_feit = train_V[rand_idx_feit]
            feed_dict_feit={X: X_batch_feit, V: V_batch_feit, isTraining: True}
            _, feit_loss_val = sess.run([train_op_feit, loss_feit],feed_dict=feed_dict_feit)

        if steps % 10==0:
            if training_style==1:
                print('gpu:', gpu_number, 'steps:', steps ,'VAE_loss:', VAE_loss_val, 'fVh_loss:', fVh_loss_val)
            elif training_style==2:
                print('gpu:', gpu_number, 'steps:', steps ,'feit_loss:', feit_loss_val)

        if steps % 1000==0:
            randpermlist_train=np.random.permutation(len(train_X))
            randpermlist_test=np.random.permutation(len(test_X))

            #Train/Test Reconstuction
            samples_train_X_recon = sess.run(inference_X_VAE_test, feed_dict={X: train_X[randpermlist_train[0:test_size]], isTraining: False})
            samples_test_X_recon = sess.run(inference_X_VAE_test, feed_dict={X: test_X[randpermlist_test[0:test_size]], isTraining: False})
            samples_train_V_recon = sess.run(inference_V_VAE_test, feed_dict={V: train_V[randpermlist_train[0:test_size]], isTraining: False})
            samples_test_V_recon = sess.run(inference_V_VAE_test, feed_dict={V: test_V[randpermlist_test[0:test_size]], isTraining: False})

            #Latent Space Walking
            z11 = sess.run(latent_VAE, feed_dict={X: test_X[randpermlist_test[0:1]], isTraining: False})
            z12 = sess.run(latent_VAE, feed_dict={X: test_X[randpermlist_test[nnn-1:nnn]], isTraining: False})
            z21 = sess.run(latent_VAE, feed_dict={X: test_X[randpermlist_test[nnn**2-nnn:nnn**2-nnn+1]], isTraining: False})
            z22 = sess.run(latent_VAE, feed_dict={X: test_X[randpermlist_test[nnn**2-1:nnn**2]], isTraining: False})

            z_latent1 = np.zeros((nnn**2,z_dim))
            T1 = np.linspace(0,1,nnn**2)
            for i1, t1 in enumerate(T1):
                z_latent1[i1,:] = z11*(1-t1) + z22*t1
            samples_latent_walk1 = sess.run(inference_Z_VAE_test, feed_dict={Z: z_latent1, isTraining: False})

            # Random Sampling
            z_rand = np.random.randn(test_size, z_dim)
            samples_random_recon = sess.run(inference_Z_VAE_test, feed_dict={Z: z_rand, isTraining: False})

            # fig11 = plot(train_X[randpermlist_train[0:nnn**2]], cmap, vmax, vmin, nnn, nnn)
            # plt.savefig(exp_dir + './out1/{}.png'.format("result_"+str(steps).zfill(3)+"_11_train_X_ori"), bbox_inches='tight', transparent=True, pad_inches=0)
            # plt.close(fig11)
            #
            # fig12 = plot(samples_train_X_recon, cmap, vmax, vmin, nnn, nnn)
            # plt.savefig(exp_dir + './out1/{}.png'.format("result_"+str(steps).zfill(3)+"_12_train_X_recon"), bbox_inches='tight', transparent=True, pad_inches=0)
            # plt.close(fig12)
            #
            # fig13 = plot(samples_train_V_recon, cmap, vmax, vmin, nnn, nnn)
            # plt.savefig(exp_dir + './out1/{}.png'.format("result_"+str(steps).zfill(3)+"_13_train_V_recon"), bbox_inches='tight', transparent=True, pad_inches=0)
            # plt.close(fig13)

            # fig21 = plot(test_X[randpermlist_test[0:nnn**2]], cmap, vmax, vmin, nnn, nnn)
            # plt.savefig(exp_dir + './out1/{}.png'.format("result_"+str(steps).zfill(3)+"_21_test_X_ori"), bbox_inches='tight', transparent=True, pad_inches=0)
            # plt.close(fig21)

            # fig22 = plot(samples_test_X_recon, cmap, vmax, vmin, nnn, nnn)
            # plt.savefig(exp_dir + './out1/{}.png'.format("result_"+str(steps).zfill(3)+"_22_test_X_recon"), bbox_inches='tight', transparent=True, pad_inches=0)
            # plt.close(fig22)

            # fig23 = plot(samples_test_V_recon, cmap, vmax, vmin, nnn, nnn)
            # plt.savefig(exp_dir + './out1/{}.png'.format("result_"+str(steps).zfill(3)+"_23_test_V_recon"), bbox_inches='tight', transparent=True, pad_inches=0)
            # plt.close(fig23)

            # fig31 = plot(samples_latent_walk1, cmap, vmax, vmin, nnn, nnn)
            # plt.savefig(exp_dir + './out1/{}.png'.format("result_"+str(steps).zfill(3)+"_31_latent_walk"), bbox_inches='tight', transparent=True, pad_inches=0)
            # plt.close(fig31)
            #
            # fig32 = plot(samples_random_recon, cmap, vmax, vmin, nnn, nnn)
            # plt.savefig(exp_dir + './out1/{}.png'.format("result_"+str(steps).zfill(3)+"_32_random_recon"), bbox_inches='tight', transparent=True, pad_inches=0)
            # plt.close(fig32)


            ### Recostruction of Selected Ventilation set
            O_inference_X_VAE_test = sess.run(inference_X_VAE_test2, feed_dict={X: feed_X, isTraining: False})
            O_inference_V_VAE_test = sess.run(inference_V_VAE_test2, feed_dict={V: feed_V, isTraining: False})

            fig41 = plot(feed_X[0:test_size2], cmap, vmax, vmin, 2, int(test_size2/2))
            plt.savefig(exp_dir + './out2/{}.png'.format("result_"+str(steps).zfill(3)+"_1_test_X_ori"), bbox_inches='tight', transparent=True, pad_inches=0)
            plt.close(fig41)

            fig42 = plot(O_inference_V_VAE_test, cmap, vmax, vmin, 2, int(test_size2/2))
            plt.savefig(exp_dir + './out2/{}.png'.format("result_"+str(steps).zfill(3)+"_2_test_V_recon"), bbox_inches='tight', transparent=True, pad_inches=0)
            plt.close(fig42)

            with h5py.File(exp_dir + './test_recon/{}.h5'.format("result_"+str(steps).zfill(3)+"_test_V_recon"), 'w') as hf:
                hf.create_dataset('EIT_test_gt',  data=feed_X[0:test_size2])
                hf.create_dataset('EIT_test_recon',  data=O_inference_V_VAE_test)

            # fig43 = plot(O_inference_X_VAE_test, cmap, vmax, vmin, 2, int(test_size2/2))
            # plt.savefig(exp_dir + './out2/{}.png'.format("result_"+str(steps).zfill(3)+"_3_test_X_recon"), bbox_inches='tight', transparent=True, pad_inches=0)
            # plt.close(fig43)

            ### Visualization of learned manifold

            ############### Case1 ###############
            ### Selected latent variable (given by encoder)
            # z_0 = sess.run(latent_VAE, feed_dict={X: feed_X[test_size2-1:test_size2], isTraining: False})
            z_0 = sess.run(latent_VAE, feed_dict={X: feed_X[test_size2-int(test_size2/2)-1:test_size2-int(test_size2/2)], isTraining: False})
            z_pertub_set = np.zeros([z_dim,2*n_delta+1,z_dim])
            for k in range(z_dim):
                z_unit_vec=np.zeros([1,z_dim])
                z_unit_vec[:,k]=1
                z_bar = z_0 - (n_delta*delta)*z_unit_vec
                for i in range(2*n_delta+1):
                    z_pertub_set[k,i,:] = z_bar + (i*delta)*z_unit_vec
            z_pertub_set = np.reshape(z_pertub_set,[z_dim*(2*n_delta+1),z_dim])
            sample_z_pertubation = sess.run(inference_Z_VAE_test3, feed_dict={Z: z_pertub_set, isTraining: False})
            sample_z_pertubation_re = np.reshape(sample_z_pertubation,[z_dim, (2*n_delta+1), input_sx*input_sy])
            ### Tangent Patch1
            sample_tangent_set = np.zeros([z_dim, (2*n_delta), input_sx*input_sy])
            for i in range(z_dim):
                for j in range(2*n_delta):
                    sample_tangent_set[i,j,:]=(sample_z_pertubation_re[i,j+1,:]-sample_z_pertubation_re[i,j,:])/delta
            sample_tangent_set = np.reshape(sample_tangent_set, [z_dim*(2*n_delta), input_sx*input_sy])

            ############### Case2 ###############
            ### zero latent vector
            z_0 = np.zeros([1,z_dim])
            z_pertub_set2 = np.zeros([z_dim,2*n_delta+1,z_dim])
            for k in range(z_dim):
                z_unit_vec=np.zeros([1,z_dim])
                z_unit_vec[:,k]=1
                z_bar = z_0 - (n_delta*delta)*z_unit_vec
                for i in range(2*n_delta+1):
                    z_pertub_set2[k,i,:] = z_bar + (i*delta)*z_unit_vec
            z_pertub_set2 = np.reshape(z_pertub_set2,[z_dim*(2*n_delta+1),z_dim])
            sample_z_pertubation2 = sess.run(inference_Z_VAE_test3, feed_dict={Z: z_pertub_set2, isTraining: False})
            sample_z_pertubation2_re = np.reshape(sample_z_pertubation2,[z_dim, (2*n_delta+1), input_sx*input_sy])

            ### Tangent Patch2
            sample_tangent_set2 = np.zeros([z_dim, (2*n_delta), input_sx*input_sy])
            for i in range(z_dim):
                for j in range(2*n_delta):
                    sample_tangent_set2[i,j,:]=(sample_z_pertubation2_re[i,j+1,:]-sample_z_pertubation2_re[i,j,:])/delta
            sample_tangent_set2 = np.reshape(sample_tangent_set2, [z_dim*(2*n_delta), input_sx*input_sy])

            tanget_scale_factor=2

            fig51 = plot(sample_z_pertubation, cmap, vmax, vmin, z_dim, 2*n_delta+1)
            plt.savefig(exp_dir + './out3/{}.png'.format("result_"+str(steps).zfill(3)+"_31_pertubation1"), bbox_inches='tight', transparent=True, pad_inches=0)
            plt.close(fig51)

            fig52 = plot(sample_tangent_set, cmap, vmax/tanget_scale_factor, vmin/tanget_scale_factor, z_dim, 2*n_delta)
            plt.savefig(exp_dir + './out3/{}.png'.format("result_"+str(steps).zfill(3)+"_32_tangent_patch1"), bbox_inches='tight', transparent=True, pad_inches=0)
            plt.close(fig52)

            fig53 = plot(sample_z_pertubation2, cmap, vmax, vmin, z_dim, 2*n_delta+1)
            plt.savefig(exp_dir + './out3/{}.png'.format("result_"+str(steps).zfill(3)+"_33_pertubation2"), bbox_inches='tight', transparent=True, pad_inches=0)
            plt.close(fig53)

            fig54 = plot(sample_tangent_set2, cmap, vmax/tanget_scale_factor, vmin/tanget_scale_factor, z_dim, 2*n_delta)
            plt.savefig(exp_dir + './out3/{}.png'.format("result_"+str(steps).zfill(3)+"_34_tangent_patch2"), bbox_inches='tight', transparent=True, pad_inches=0)
            plt.close(fig54)

            save_path = saver.save(sess, exp_dir + "./"+training_name+"_training_weight_"+exp_date+'_'+exp_num+".ckpt", global_step=steps)
            print("Model saved in file: %s" % save_path)

            #Visualization of Manifold with 2D Latent Space
            if z_dim==2:
                if not os.path.exists(exp_dir+'./out4/'):
                    os.makedirs(exp_dir+'./out4/')

                z2d_val=3
                # z2d_val=1.5

                z2d_11 = np.array([[-z2d_val,z2d_val]])
                z2d_12 = np.array([[z2d_val,z2d_val]])
                z2d_21 = np.array([[-z2d_val,-z2d_val]])
                z2d_22 = np.array([[z2d_val,-z2d_val]])

                z2d_L = np.zeros((grid_num_2d,z_dim))
                z2d_R = np.zeros((grid_num_2d,z_dim))
                T_2d = np.linspace(0,1,grid_num_2d)

                for i2, t2 in enumerate(T_2d):
                    z2d_L[i2,:] = z2d_11*(1-t2) + z2d_21*t2
                    z2d_R[i2,:] = z2d_12*(1-t2) + z2d_22*t2
                z2d_latent=np.zeros((grid_num_2d**2,z_dim))
                for k in range(grid_num_2d):
                    for j, t3 in enumerate(T_2d):
                        z2d_latent[k*grid_num_2d+j,:] = z2d_L[k,:]*(1-t3) + z2d_R[k,:]*t3

                samples_latent_2d = sess.run(inference_Z_VAE_test_2d, feed_dict={Z: z2d_latent, isTraining: False})
                fig61 = plot(samples_latent_2d, cmap, vmax, vmin, grid_num_2d, grid_num_2d)
                plt.savefig(exp_dir + './out4/{}.png'.format("result_"+str(steps).zfill(3)+"_latent_2d"), bbox_inches='tight', transparent=True, pad_inches=0)
                plt.close(fig61)

        #tensorboard part
        if training_style==1:
            summary_loss_VAE = sess.run(summary_loss_op_VAE,feed_dict={loss_steps_VAE: VAE_loss_val})
            summary_loss_fVh = sess.run(summary_loss_op_fVh,feed_dict={loss_steps_fVh: fVh_loss_val})
            writer.add_summary(summary_loss_VAE, global_step = steps)
            writer.add_summary(summary_loss_fVh, global_step = steps)
        elif training_style==2:
            summary_loss_feit = sess.run(summary_loss_op_feit,feed_dict={loss_steps_feit: feit_loss_val})
            writer.add_summary(summary_loss_feit, global_step = steps)


    print("Learning finished!")
