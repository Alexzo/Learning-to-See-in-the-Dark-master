
#uniform content loss + adaptive threshold + per_class_input + recursive G
#improvement upon cqf37
from __future__ import division
import os,time,scipy.io
import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.contrib.layers.python.layers import initializers
import numpy as np
import pdb
import rawpy
import glob



def pack_raw(raw):
    #pack Bayer image to 4 channels
    im = raw.raw_image_visible.astype(np.float32)
    im = np.maximum(im - 64, 0)/ (1023 - 64) #subtract the black level

    im = np.expand_dims(im,axis=2)
    img_shape = im.shape
    H = img_shape[0]
    W = img_shape[1]

    out = np.concatenate((im[0:H:2,0:W:2,:],
                       im[0:H:2,1:W:2,:],
                       im[1:H:2,1:W:2,:],
                       im[1:H:2,0:W:2,:]), axis=2)
    return out


path = './dataset/raw_data/0000.dng'
checkpoint_dir = './checkpoint/Sony/'
result_dir = './checkpoint/Sony/'
input_dir = './dataset/seattle_night_raw_only/'
gt_dir = './dataset/seattle_night_raw_only/'
# raw=rawpy.imread(path)
# out=pack_raw(raw)
def network(input):
    conv1=slim.conv2d(input,32,[3,3], rate=1, activation_fn=lrelu,scope='g_conv1_1')
    conv1=slim.conv2d(conv1,32,[3,3], rate=1, activation_fn=lrelu,scope='g_conv1_2')
    pool1=slim.max_pool2d(conv1, [2, 2], padding='SAME' )

    conv2=slim.conv2d(pool1,64,[3,3], rate=1, activation_fn=lrelu,scope='g_conv2_1')
    conv2=slim.conv2d(conv2,64,[3,3], rate=1, activation_fn=lrelu,scope='g_conv2_2')
    pool2=slim.max_pool2d(conv2, [2, 2], padding='SAME' )

    conv3=slim.conv2d(pool2,128,[3,3], rate=1, activation_fn=lrelu,scope='g_conv3_1')
    conv3=slim.conv2d(conv3,128,[3,3], rate=1, activation_fn=lrelu,scope='g_conv3_2')
    pool3=slim.max_pool2d(conv3, [2, 2], padding='SAME' )

    conv4=slim.conv2d(pool3,256,[3,3], rate=1, activation_fn=lrelu,scope='g_conv4_1')
    conv4=slim.conv2d(conv4,256,[3,3], rate=1, activation_fn=lrelu,scope='g_conv4_2')
    pool4=slim.max_pool2d(conv4, [2, 2], padding='SAME' )

    conv5=slim.conv2d(pool4,512,[3,3], rate=1, activation_fn=lrelu,scope='g_conv5_1')
    conv5=slim.conv2d(conv5,512,[3,3], rate=1, activation_fn=lrelu,scope='g_conv5_2')

    up6 =  upsample_and_concat( conv5, conv4, 256, 512  )
    conv6=slim.conv2d(up6,  256,[3,3], rate=1, activation_fn=lrelu,scope='g_conv6_1')
    conv6=slim.conv2d(conv6,256,[3,3], rate=1, activation_fn=lrelu,scope='g_conv6_2')

    up7 =  upsample_and_concat( conv6, conv3, 128, 256  )
    conv7=slim.conv2d(up7,  128,[3,3], rate=1, activation_fn=lrelu,scope='g_conv7_1')
    conv7=slim.conv2d(conv7,128,[3,3], rate=1, activation_fn=lrelu,scope='g_conv7_2')

    up8 =  upsample_and_concat( conv7, conv2, 64, 128 )
    conv8=slim.conv2d(up8,  64,[3,3], rate=1, activation_fn=lrelu,scope='g_conv8_1')
    conv8=slim.conv2d(conv8,64,[3,3], rate=1, activation_fn=lrelu,scope='g_conv8_2')

    up9 =  upsample_and_concat( conv8, conv1, 32, 64 )
    conv9=slim.conv2d(up9,  32,[3,3], rate=1, activation_fn=lrelu,scope='g_conv9_1')
    conv9=slim.conv2d(conv9,32,[3,3], rate=1, activation_fn=lrelu,scope='g_conv9_2')

    conv10=slim.conv2d(conv9,12,[1,1], rate=1, activation_fn=None, scope='g_conv10')
    out = tf.depth_to_space(conv10,2)
    return out

def lrelu(x):
    return tf.maximum(x*0.2,x)

def upsample_and_concat(x1, x2, output_channels, in_channels):

    pool_size = 2
    deconv_filter = tf.Variable(tf.truncated_normal( [pool_size, pool_size, output_channels, in_channels], stddev=0.02))
    deconv = tf.nn.conv2d_transpose(x1, deconv_filter, tf.shape(x2) , strides=[1, pool_size, pool_size, 1] )

    deconv_output =  tf.concat([deconv, x2],3)
    deconv_output.set_shape([None, None, None, output_channels*2])

    return deconv_output

#get train and test IDs
train_fns = glob.glob(gt_dir + '0*.dng')
train_ids = []
for i in range(len(train_fns)):
    _, train_fn = os.path.split(train_fns[i])
    train_ids.append((train_fns[i],int(train_fn[0:4])))

test_fns = glob.glob(gt_dir + '0*.dng')
test_ids = []
for i in range(len(test_fns)):
    _, test_fn = os.path.split(test_fns[i])
    test_ids.append((test_fns[i],int(test_fn[0:4])))

# for file in train_ids:
#     print(file[0])
sess=tf.Session()
in_image=tf.placeholder(tf.float32,[None,None,None,4])
gt_image=tf.placeholder(tf.float32,[None,None,None,3])
out_image=network(in_image)

saver=tf.train.Saver()
sess.run(tf.global_variables_initializer())
ckpt=tf.train.get_checkpoint_state(checkpoint_dir)
if ckpt:
    print('loaded '+ckpt.model_checkpoint_path)
    saver.restore(sess,ckpt.model_checkpoint_path)

if not os.path.isdir(result_dir + 'final2/'):
    os.makedirs(result_dir + 'final2/')
fildir=('./dataset/seattle_night_raw_only/0175.dng', 175)
train_ids=[fildir]
for fildir in train_ids:
    gt_path=fildir[0]
    raw = rawpy.imread(gt_path)
    ratio=1.0
    input_full = np.expand_dims(pack_raw(raw), axis=0)*ratio

    im = raw.postprocess(use_camera_wb=True, half_size=False, no_auto_bright=True, output_bps=16)
    # scale_full = np.expand_dims(np.float32(im/65535.0),axis = 0)*ratio
    scale_full = np.expand_dims(np.float32(im / 65535.0), axis=0)

    gt_raw = rawpy.imread(gt_path)
    im = gt_raw.postprocess(use_camera_wb=True, half_size=False, no_auto_bright=True, output_bps=16)
    gt_full = np.expand_dims(np.float32(im / 65535.0), axis=0)

    input_full = np.minimum(input_full, 1.0)

    output = sess.run(out_image, feed_dict={in_image: input_full})
    output = np.minimum(np.maximum(output, 0), 1)

    output = output[0, :, :, :]
    gt_full = gt_full[0, :, :, :]
    scale_full = scale_full[0, :, :, :]
    scale_full = scale_full * np.mean(gt_full) / np.mean(
    scale_full)  # scale the low-light image to the same mean of the groundtruth

    scipy.misc.toimage(output * 255, high=255, low=0, cmin=0, cmax=255).save(
            result_dir + 'final2/%5d_00_%d_out.png' % (fildir[1], ratio))
    # scipy.misc.toimage(scale_full * 255, high=255, low=0, cmin=0, cmax=255).save(
    #         result_dir + 'final2/%5d_00_%d_scale.png' % (fildir[1], 3))
    # scipy.misc.toimage(gt_full * 255, high=255, low=0, cmin=0, cmax=255).save(
    #         result_dir + 'final2/%5d_00_%d_gt.png' % (fildir[1], 3))

# for test_id in test_ids:
#     # test the first image in each sequence
#     in_files = glob.glob(input_dir + '%.dng' % test_id)
#     for k in range(len(in_files)):
#         in_path = in_files[k]
#         _, in_fn = os.path.split(in_path)
#         print(in_fn)
#         gt_files = glob.glob(gt_dir + '%.dng' % test_id)
#         gt_path = gt_files[0]
#         # _, gt_fn = os.path.split(gt_path)
#         # in_exposure = float(in_fn[9:-5])
#         # gt_exposure = float(gt_fn[9:-5])
#         # ratio = min(gt_exposure / in_exposure, 300)
#
#         raw = rawpy.imread(in_path)
#         input_full = np.expand_dims(pack_raw(raw), axis=0) * 1.0
#
#         im = raw.postprocess(use_camera_wb=True, half_size=False, no_auto_bright=True, output_bps=16)
#         # scale_full = np.expand_dims(np.float32(im/65535.0),axis = 0)*ratio
#         scale_full = np.expand_dims(np.float32(im / 65535.0), axis=0)
#
#         gt_raw = rawpy.imread(gt_path)
#         im = gt_raw.postprocess(use_camera_wb=True, half_size=False, no_auto_bright=True, output_bps=16)
#         gt_full = np.expand_dims(np.float32(im / 65535.0), axis=0)
#
#         input_full = np.minimum(input_full, 1.0)
#
#         output = sess.run(out_image, feed_dict={in_image: input_full})
#         output = np.minimum(np.maximum(output, 0), 1)
#
#         output = output[0, :, :, :]
#         gt_full = gt_full[0, :, :, :]
#         scale_full = scale_full[0, :, :, :]
#         scale_full = scale_full * np.mean(gt_full) / np.mean(
#             scale_full)  # scale the low-light image to the same mean of the groundtruth
#
#         scipy.misc.toimage(output * 255, high=255, low=0, cmin=0, cmax=255).save(
#             result_dir + 'final/%5d_00_%d_out.png' % (test_id, 1))
#         scipy.misc.toimage(scale_full * 255, high=255, low=0, cmin=0, cmax=255).save(
#             result_dir + 'final/%5d_00_%d_scale.png' % (test_id, 1))
#         scipy.misc.toimage(gt_full * 255, high=255, low=0, cmin=0, cmax=255).save(
#             result_dir + 'final/%5d_00_%d_gt.png' % (test_id, 1))