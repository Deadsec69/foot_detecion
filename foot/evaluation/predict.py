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
import pandas as pd

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

if __name__ == '__main__':
    model = MobileNetv2((224, 224, 3), 1+1+4+4, 1.0)
    model.load_weights('4my_model.h5')
    
    lr_df = pd.read_csv("lr_foot.csv")
    
    
    url = "https://newvalues.s3-us-west-1.amazonaws.com/IMG_3433.JPG"

    image = Image.open(urllib.request.urlopen(url))
    img_512 = cv2.cvtColor(numpy.array(image), cv2.COLOR_RGB2BGR)
    img=cv2.resize(img_512, (224,224), interpolation = cv2.INTER_AREA)
    cv2.imwrite("sample.jpg", img)
    img = numpy.expand_dims(img, axis=0)
    arr = model.predict(img)
   
    print(arr)
    xcl=arr[0][2]
    ycl=arr[0][3]
    widl=arr[0][4]
    heil=arr[0][5]
   
    xcr=arr[0][6]
    ycr=arr[0][7]
    widr=arr[0][8]
    heir=arr[0][9]
    z = 224 

    xminl=int((xcl-(widl/2))*z)
    xmaxl=int((xcl+(widl/2))*z)
    yminl=int((ycl-(heil/2))*z)
    ymaxl=int((ycl+(heil/2))*z)
	
    xminr=int((xcr-(widr/2))*z)
    xmaxr=int((xcr+(widr/2))*z)
    yminr=int((ycr-(heir/2))*z)
    ymaxr=int((ycr+(heir/2))*z)
    
    print(xminl,xmaxl,yminl, ymaxl, ': ')
    print(xminr,xmaxr,yminr, ymaxr, ': ')
    
    img = cv2.rectangle(img[0],(xminl,yminl),(xmaxl,ymaxl),(255,0,0), 2)
    img = cv2.rectangle(img,(xminr,yminr),(xmaxr,ymaxr),(255,0,0), 2)	
   
    cv2.imwrite("predict.jpg", img)
    print(arr)

