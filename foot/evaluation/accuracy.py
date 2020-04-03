from keras.models import Model
from keras.layers import Input, Conv2D, GlobalAveragePooling2D, Dropout
from keras.layers import Activation, BatchNormalization, Add, Reshape, DepthwiseConv2D
from keras.utils.vis_utils import plot_model
from keras.activations import sigmoid
from keras import backend as K
from PIL import Image 
import cv2
import urllib.request 
import numpy 
import numpy as np
import pandas as pd
import os

def _make_divisible(v, divisor, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def relu6(x):
    """Relu 6
    """
    return K.relu(x, max_value=6.0)


def _conv_block(inputs, filters, kernel, strides):

    channel_axis = 1 if K.image_data_format() == 'channels_first' else -1

    x = Conv2D(filters, kernel, padding='same', strides=strides)(inputs)
    x = BatchNormalization(axis=channel_axis)(x)
    return Activation(relu6)(x)


def _bottleneck(inputs, filters, kernel, t, alpha, s, r=False):

    channel_axis = 1 if K.image_data_format() == 'channels_first' else -1
    # Depth
    tchannel = K.int_shape(inputs)[channel_axis] * t
    # Width
    cchannel = int(filters * alpha)

    x = _conv_block(inputs, tchannel, (1, 1), (1, 1))

    x = DepthwiseConv2D(kernel, strides=(s, s), depth_multiplier=1, padding='same')(x)
    x = BatchNormalization(axis=channel_axis)(x)
    x = Activation(relu6)(x)

    x = Conv2D(cchannel, (1, 1), strides=(1, 1), padding='same')(x)
    x = BatchNormalization(axis=channel_axis)(x)

    if r:
        x = Add()([x, inputs])

    return x


def _inverted_residual_block(inputs, filters, kernel, t, alpha, strides, n):

    x = _bottleneck(inputs, filters, kernel, t, alpha, strides)

    for i in range(1, n):
        x = _bottleneck(x, filters, kernel, t, alpha, 1, True)

    return x


def MobileNetv2(input_shape, k, alpha=1.0):
    
    inputs = Input(shape=input_shape)

    first_filters = _make_divisible(32 * alpha, 8)
    
    x = _conv_block(inputs, first_filters, (3, 3), strides=(2, 2))

    x = _inverted_residual_block(x, 16, (3, 3), t=1, alpha=alpha, strides=1, n=1)
    x = _inverted_residual_block(x, 24, (3, 3), t=6, alpha=alpha, strides=2, n=2)
    x = _inverted_residual_block(x, 32, (3, 3), t=6, alpha=alpha, strides=2, n=3)
    x = _inverted_residual_block(x, 64, (3, 3), t=6, alpha=alpha, strides=2, n=4)
    x = _inverted_residual_block(x, 96, (3, 3), t=6, alpha=alpha, strides=1, n=3)
    x = _inverted_residual_block(x, 160, (3, 3), t=6, alpha=alpha, strides=2, n=3)
    x = _inverted_residual_block(x, 320, (3, 3), t=6, alpha=alpha, strides=1, n=1)

    if alpha > 1.0:
        last_filters = _make_divisible(1280 * alpha, 8)
    else:
        last_filters = 1280

    x = _conv_block(x, last_filters, (1, 1), strides=(1, 1))
    
    print("shape: ", x.shape)
    
    x = GlobalAveragePooling2D()(x)
    x = Reshape((1, 1, last_filters))(x)
    x = Dropout(0.3, name='Dropout')(x)
    x = Conv2D(k, (1, 1), padding='same')(x)
    
#     x = Activation('softmax', name='softmax')(x)
    output = Reshape((k,))(x)

    model = Model(inputs, output)
    # plot_model(model, to_file='images/MobileNetv2.png', show_shapes=True)

    return model


def custom_loss(ytrue, ypred):
    
    true_box_prob = ytrue[... , :2]
    true_box_coords1 = ytrue[... , 2:6]
    true_box_coords2 = ytrue[... , 6:10]
    
    pred_box_prob = tf.sigmoid(ypred[... , :2])
    pred_box_coords1 = ypred[... , 2:6]
    pred_box_coords2 = ypred[... , 6:10]
    
    r1= tf.keras.losses.mse(y_true=true_box_coords1,y_pred=pred_box_coords1)
    r2= tf.keras.losses.mse(y_true=true_box_coords2,y_pred=pred_box_coords2)
    
    r1 = tf.multiply(r1 ,true_box_prob[:,0])
    r2 = tf.multiply(r2 ,true_box_prob[:,1])
    
    classification_loss = tf.keras.losses.binary_crossentropy(y_true=true_box_prob,y_pred=pred_box_prob)
    
    return (r1+r2)/1000.0*(1) + classification_loss

def bb_intersection_over_union(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)
    # return the intersection over union value
    return  (xA, yA, xB, yB), iou

if __name__ == '__main__':
    model = MobileNetv2((224, 224, 3), 1+1+4+4, 1.0)
    model.load_weights('4my_model.h5')
    
    lr_df = pd.read_csv("lr_foot.csv")
    
    # x_images = []
    y_gt_left = []
    y_gt_right = []

    images_in_dir = os.listdir('jpg_images')

    count_right = 0
    count_total = 0
    iou_threshold = 0.5

    for idx,row_obj in lr_df.iterrows():
        #if idx > 100:
        #    continue
        print(idx)        

        if row_obj.filename not in images_in_dir:
            continue
        
        img_512 = cv2.imread("jpg_images/" + str(row_obj.filename))
        img_224=cv2.resize(img_512, (224,224), interpolation = cv2.INTER_AREA)
        
        # x_images.append(img)
        img_224 = numpy.expand_dims(img_224, axis=0)


        y_gt_left = ([row_obj.lxmin,row_obj.lymin, row_obj.lxmax, row_obj.lymax])
        y_gt_right = ([row_obj.rxmin,row_obj.rymin, row_obj.rxmax, row_obj.rymax])

        pred_arr = model.predict(img_224)

        lpred_prob = 1/(1 + np.exp(pred_arr[0][0])) > 0        
        rpred_prob = 1/(1 + np.exp(pred_arr[0][1])) > 0

        lactual_prob = row_obj.lxmin and row_obj.lymin and row_obj.lxmax and row_obj.lymax
        ractual_prob = row_obj.rxmin and row_obj.rymin and row_obj.rxmax and row_obj.rymax

        z = 512     

        xcl=pred_arr[0][2]
        ycl=pred_arr[0][3]
        widl=pred_arr[0][4]
        heil=pred_arr[0][5]

        xcr=pred_arr[0][6]
        ycr=pred_arr[0][7]
        widr=pred_arr[0][8]
        heir=pred_arr[0][9]

        xminl=int((xcl-(widl/2))*z)
        xmaxl=int((xcl+(widl/2))*z)
        yminl=int((ycl-(heil/2))*z)
        ymaxl=int((ycl+(heil/2))*z)
        
        xminr=int((xcr-(widr/2))*z)
        xmaxr=int((xcr+(widr/2))*z)
        yminr=int((ycr-(heir/2))*z)
        ymaxr=int((ycr+(heir/2))*z)

        if lpred_prob and lactual_prob:
            # print("lactual: ",row_obj.lxmin,row_obj.lymin, row_obj.lxmax, row_obj.lymax)
            # print("lpred: ",xminl,yminl,ymaxl,ymaxr)

            _, iou_l = bb_intersection_over_union([row_obj.lxmin,row_obj.lymin, row_obj.lxmax, row_obj.lymax], [xminl,yminl,xmaxl,ymaxl])
            
            #print("iou_l: ", iou_l)          

            if(iou_l > iou_threshold):
                count_right+=1

            # cv2.rectangle(img_512,(xminl,yminl),(xmaxl,ymaxl),(255,0,0), 2)
            # cv2.rectangle(img_512,(row_obj.lxmin,row_obj.lymin), (row_obj.lxmax, row_obj.lymax),(0,0,255), 2)
            # cv2.imwrite("prediction.jpg", img_512)

        
        elif rpred_prob and ractual_prob:
            # print("ractual: ",row_obj.rxmin,row_obj.rymin, row_obj.rxmax, row_obj.rymax)
            # print("rpred: ",xminr,yminr,xmaxr,ymaxr)
            _, iou_r = bb_intersection_over_union([row_obj.rxmin,row_obj.rymin, row_obj.rxmax, row_obj.rymax], [xminr,yminr,xmaxr,ymaxr])  
            
            #print("iou_r: ", iou_r)          
            
            if(iou_r > iou_threshold):
                count_right += 1

            # cv2.rectangle(img_512,(xminr,yminr),(xmaxr,ymaxr),(255,0,0), 2)            
            # cv2.rectangle(img_512,(row_obj.rxmin,row_obj.rymin), (row_obj.rxmax, row_obj.rymax),(0,0,255), 2)
            # cv2.rectangle(img_512,(xA, yA),(xB, yB),(0,255,0), 2)            
            
            # cv2.imwrite("prediction.jpg", img_512)
            

        count_total += 1 
    print("accuracy: ", count_right/count_total, count_right, count_total )
    
    # print(xminl,xmaxl,yminl, ymaxl, ': ')
    # print(xminr,xmaxr,yminr, ymaxr, ': ')
    
    # img = cv2.rectangle(img[0],(xminl,yminl),(xmaxl,ymaxl),(255,0,0), 2)
    # img = cv2.rectangle(img,(xminr,yminr),(xmaxr,ymaxr),(255,0,0), 2)	

    # cv2.imwrite("predlsict.jpg", img)
    # print(arr)
