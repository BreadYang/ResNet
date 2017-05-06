from __future__ import print_function

import numpy as np
import warnings
import h5py
import os
from keras.layers import Input
from keras import layers
from keras.layers import Dense
from keras.layers import Activation
from keras.layers import Flatten
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import GlobalMaxPooling2D
from keras.layers import ZeroPadding2D
from keras.layers import AveragePooling2D
from keras.layers import GlobalAveragePooling2D
from keras.layers import BatchNormalization
from keras.models import Model
from keras.preprocessing import image
import keras.backend as K
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras.applications.imagenet_utils import decode_predictions
from keras.applications.imagenet_utils import preprocess_input
from keras.applications.imagenet_utils import _obtain_input_shape
from keras.engine.topology import get_source_inputs
from keras.utils.io_utils import HDF5Matrix
from keras import optimizers

from attention_layers import *



from image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, History
import json


WEIGHTS_PATH = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.2/resnet50_weights_tf_dim_ordering_tf_kernels.h5'
# WEIGHTS_PATH_NO_TOP = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.2/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'

WEIGHTS_PATH_NO_TOP = 'resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'

def identity_block(connected, input_tensor, kernel_size, filters, stage, block):
    """The identity block is the block that has no conv layer at shortcut.
    # Arguments
        input_tensor: input tensor
        kernel_size: defualt 3, the kernel size of middle conv layer at main path
        filters: list of integers, the filterss of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
    # Returns
        Output tensor for the block.
    """
    filters1, filters2, filters3 = filters
    if K.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = Conv2D(filters1, (1, 1), name=conv_name_base + '2a')(input_tensor)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters2, kernel_size,
               padding='same', name=conv_name_base + '2b')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters3, (1, 1), name=conv_name_base + '2c')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

    #Identity connection
    if connected == 1:
        x = layers.add([x, input_tensor])
    #Softmax attention
    elif connected == 2:
        softmax_att = Lambda(softmaxLayer,output_shape=softmaxLayer_output_shape)(x)
        att_x = layers.multiply([x,softmax_att])
        att_input_tensor = layers.multiply([input_tensor,softmax_att])
        x = layers.add([att_x,att_input_tensor])
    #Sum of absolute values
    elif connected == 3:
        abs_val_att = Lambda(sumAbsVal,output_shape=sumAbsVal_output_shape)(x)
        att_x = layers.multiply([x,abs_val_att])
        att_input_tensor = layers.multiply([input_tensor,abs_val_att])
        x = layers.add([att_x, att_input_tensor])
    #Maxout
    elif connected == 4:
        maxout_att = Lambda(maxoutLayer,output_shape=maxoutLayer_output_shape)(x)
        att_x = layers.multiply([x,maxout_att])
        att_input_tensor = layers.multiply([input_tensor,maxout_att])
        x = layers.add([att_x, att_input_tensor])
    x = Activation('relu')(x)
    return x


def conv_block(connected, input_tensor, kernel_size, filters, stage, block, strides=(2, 2)):
    """conv_block is the block that has a conv layer at shortcut
    # Arguments
        input_tensor: input tensor
        kernel_size: defualt 3, the kernel size of middle conv layer at main path
        filters: list of integers, the filterss of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
    # Returns
        Output tensor for the block.
    Note that from stage 3, the first conv layer at main path is with strides=(2,2)
    And the shortcut should have strides=(2,2) as well
    """
    filters1, filters2, filters3 = filters
    if K.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = Conv2D(filters1, (1, 1), strides=strides,
               name=conv_name_base + '2a')(input_tensor)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters2, kernel_size, padding='same',
               name=conv_name_base + '2b')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters3, (1, 1), name=conv_name_base + '2c')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

    shortcut = Conv2D(filters3, (1, 1), strides=strides,
                      name=conv_name_base + '1')(input_tensor)
    shortcut = BatchNormalization(axis=bn_axis, name=bn_name_base + '1')(shortcut)

    if connected == 1:
        x = layers.add([x, shortcut])
    x = Activation('relu')(x)
    return x


def ResNet50(state, weights, include_top=True,
             input_tensor=None, input_shape=None,
             pooling=None,
             classes=200):
    """Instantiates the ResNet50 architecture.
    Optionally loads weights pre-trained
    on ImageNet. Note that when using TensorFlow,
    for best performance you should set
    `image_data_format="channels_last"` in your Keras config
    at ~/.keras/keras.json.
    The model and the weights are compatible with both
    TensorFlow and Theano. The data format
    convention used by the model is the one
    specified in your Keras config file.
    # Arguments
        state: [0,1,0,1...] vector indicates the connection
            of identity passing
        include_top: whether to include the fully-connected
            layer at the top of the network.
        weights: one of `None` (random initialization)
            or "imagenet" (pre-training on ImageNet).
        input_tensor: optional Keras tensor (i.e. output of `layers.Input()`)
            to use as image input for the model.
        input_shape: optional shape tuple, only to be specified
            if `include_top` is False (otherwise the input shape
            has to be `(224, 224, 3)` (with `channels_last` data format)
            or `(3, 224, 244)` (with `channels_first` data format).
            It should have exactly 3 inputs channels,
            and width and height should be no smaller than 197.
            E.g. `(200, 200, 3)` would be one valid value.
        pooling: Optional pooling mode for feature extraction
            when `include_top` is `False`.
            - `None` means that the output of the model will be
                the 4D tensor output of the
                last convolutional layer.
            - `avg` means that global average pooling
                will be applied to the output of the
                last convolutional layer, and thus
                the output of the model will be a 2D tensor.
            - `max` means that global max pooling will
                be applied.
        classes: optional number of classes to classify images
            into, only to be specified if `include_top` is True, and
            if no `weights` argument is specified.
    # Returns
        A Keras model instance.
    # Raises
        ValueError: in case of invalid argument for `weights`,
            or invalid input shape.
    """
    if weights not in {'imagenet', 'current'}:
        raise ValueError('The `weights` argument should be either '
                         '`None` (random initialization) or `imagenet` '
                         '(pre-training on ImageNet).')

    if weights == 'imagenet' and include_top and classes != 200:
        raise ValueError('If using `weights` as imagenet with `include_top`'
                         ' as true, `classes` should be 200')

    # Determine proper input shape
    input_shape = _obtain_input_shape(input_shape,
                                      default_size=224,
                                      min_size=197,
                                      data_format=K.image_data_format(),
                                      include_top=include_top)

    if input_tensor is None:
        img_input = Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(input_tensor):
            img_input = Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor
    if K.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1

    x = ZeroPadding2D((3, 3))(img_input)
    x = Conv2D(64, (7, 7), strides=(2, 2), name='conv1')(x)
    x = BatchNormalization(axis=bn_axis, name='bn_conv1')(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)

    x = conv_block(state[0], x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1))
    x = identity_block(state[1], x, 3, [64, 64, 256], stage=2, block='b')
    x = identity_block(state[2], x, 3, [64, 64, 256], stage=2, block='c')

    x = conv_block(state[3], x, 3, [128, 128, 512], stage=3, block='a')
    x = identity_block(state[4], x, 3, [128, 128, 512], stage=3, block='b')
    x = identity_block(state[5], x, 3, [128, 128, 512], stage=3, block='c')
    x = identity_block(state[6], x, 3, [128, 128, 512], stage=3, block='d')

    x = conv_block(state[7], x, 3, [256, 256, 1024], stage=4, block='a')
    x = identity_block(state[8], x, 3, [256, 256, 1024], stage=4, block='b')
    x = identity_block(state[9], x, 3, [256, 256, 1024], stage=4, block='c')
    x = identity_block(state[10], x, 3, [256, 256, 1024], stage=4, block='d')
    x = identity_block(state[11], x, 3, [256, 256, 1024], stage=4, block='e')
    x = identity_block(state[12], x, 3, [256, 256, 1024], stage=4, block='f')

    x = conv_block(state[13], x, 3, [512, 512, 2048], stage=5, block='a')
    x = identity_block(state[14], x, 3, [512, 512, 2048], stage=5, block='b')
    x = identity_block(state[15], x, 3, [512, 512, 2048], stage=5, block='c')

    x = AveragePooling2D((7, 7), name='avg_pool')(x)

    if include_top:
        x = Flatten()(x)
        x = Dense(classes, activation='softmax', name='fc200')(x)
    else:
        if pooling == 'avg':
            x = GlobalAveragePooling2D()(x)
        elif pooling == 'max':
            x = GlobalMaxPooling2D()(x)

    # Ensure that the model takes into account
    # any potential predecessors of `input_tensor`.
    if input_tensor is not None:
        inputs = get_source_inputs(input_tensor)
    else:
        inputs = img_input
    # Create model.
    model = Model(inputs, x, name='resnet50')
    model.load_weights(WEIGHTS_PATH_NO_TOP, by_name=True)
    # model.load_weights(weights_path)
    # load weights
    if weights == 'imagenet':
        # model.load_weights('current_model.h5')

        if K.backend() == 'theano':
            layer_utils.convert_all_kernels_in_model(model)

        if K.image_data_format() == 'channels_first':
            if include_top:
                maxpool = model.get_layer(name='avg_pool')
                shape = maxpool.output_shape[1:]
                dense = model.get_layer(name='fc200')
                layer_utils.convert_dense_weights_data_format(dense, shape, 'channels_first')

            if K.backend() == 'tensorflow':
                warnings.warn('You are using the TensorFlow backend, yet you '
                              'are using the Theano '
                              'image data format convention '
                              '(`image_data_format="channels_first"`). '
                              'For best performance, set '
                              '`image_data_format="channels_last"` in '
                              'your Keras config '
                              'at ~/.keras/keras.json.')

    elif weights == 'current':
        pass
        # model.load_weights('current_model.h5')
    return model


# To be called only by save_bottleneck_features
def bottom_model(state, input_shape=None, weights_path=WEIGHTS_PATH_NO_TOP):
    input_shape = _obtain_input_shape(input_shape,
                                      default_size=224,
                                      min_size=197,
                                      data_format=K.image_data_format(),
                                      include_top=True)


    img_input = Input(shape=input_shape)

    if K.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1

    x = ZeroPadding2D((3, 3))(img_input)
    x = Conv2D(64, (7, 7), strides=(2, 2), name='conv1')(x)
    x = BatchNormalization(axis=bn_axis, name='bn_conv1')(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)

    x = conv_block(state[0], x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1))
    x = identity_block(state[1], x, 3, [64, 64, 256], stage=2, block='b')
    x = identity_block(state[2], x, 3, [64, 64, 256], stage=2, block='c')

    x = conv_block(state[3], x, 3, [128, 128, 512], stage=3, block='a')
    x = identity_block(state[4], x, 3, [128, 128, 512], stage=3, block='b')
    x = identity_block(state[5], x, 3, [128, 128, 512], stage=3, block='c')
    x = identity_block(state[6], x, 3, [128, 128, 512], stage=3, block='d')

    x = conv_block(state[7], x, 3, [256, 256, 1024], stage=4, block='a')


    inputs = img_input
    # Create model.
    model = Model(inputs, x, name='resnet50')
    # MAKE SURE NAMES MATCH
    model.load_weights(weights_path, by_name=True)
    return model

# Train only top model
# State before index 8 is ignored since it is part of bottom model
# If pretrained_weights not given loads from save_load_weights
# always saves to save_load_weights when val loss reduces
def top_model(state, classes=200, weights_path='top_half_weights.f5'):
    # Build bottom half
    print("top_model started")

    inputs = Input(shape=(14, 14, 1024))

    x = identity_block(state[8], inputs, 3, [256, 256, 1024], stage=4, block='b')
    x = identity_block(state[9], x, 3, [256, 256, 1024], stage=4, block='c')
    x = identity_block(state[10], x, 3, [256, 256, 1024], stage=4, block='d')
    x = identity_block(state[11], x, 3, [256, 256, 1024], stage=4, block='e')
    x = identity_block(state[12], x, 3, [256, 256, 1024], stage=4, block='f')

    x = conv_block(state[13], x, 3, [512, 512, 2048], stage=5, block='a')
    x = identity_block(state[14], x, 3, [512, 512, 2048], stage=5, block='b')
    x = identity_block(state[15], x, 3, [512, 512, 2048], stage=5, block='c')

    x = AveragePooling2D((7, 7), name='avg_pool')(x)

    x = Flatten()(x)
    x = Dense(classes, activation='softmax', name='fc200')(x)
    model = Model(inputs, x, name='resnet50')
    if weights_path != None:
        model.load_weights(weights_path, by_name=True)


    model.compile(optimizer=optimizers.SGD(lr=5e-4, momentum=0.0),
                    loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    print("top_model compiled")
    return model
#

def finetune_top_model(model, save_weights='top_half_weights.f5', epochs=10, batch_size=64, mini_batch=True):
    h5f = h5py.File('bottleneck_features.h5','r')

    # If you want to load into memory do the following
    # train_data = h5f['training'][:5000]
    # train_labels = h5f['train_labels'][:5000]

    train_data = HDF5Matrix("bottleneck_features.h5", 'training')
    train_labels = HDF5Matrix("bottleneck_features.h5", 'train_labels')

    val_data = HDF5Matrix("bottleneck_features.h5", 'val')
    val_labels = HDF5Matrix("bottleneck_features.h5", 'val_labels')
    checkpointer = ModelCheckpoint(filepath=save_weights, verbose=1, save_best_only=True)
    history = History()

    # If mini_batch then we split datasets into 10 parts and pick a 1/10th training
    # and 1/10th validation set for 10 iterations



    if mini_batch:
        pick = np.random.randint(0,9)
        train_data = HDF5Matrix("bottleneck_features.h5", 'training', start=10000*pick, end=10000*(pick+1))
        train_labels = HDF5Matrix("bottleneck_features.h5", 'train_labels', start=10000*pick, end=10000*(pick+1))
        pick = np.random.randint(0,9)
        val_data = HDF5Matrix("bottleneck_features.h5", 'val', start=1000*pick, end=1000*(pick+1))
        val_labels = HDF5Matrix("bottleneck_features.h5", 'val_labels', start=1000*pick, end=1000*(pick+1))
    try:
        history_returned = model.fit(train_data, train_labels, validation_data=(val_data,val_labels),epochs=epochs, batch_size=batch_size, shuffle='batch', callbacks=[checkpointer, history])
        return history_returned
    except KeyboardInterrupt as e:
        print("keyboard interrupted, saving to history.json and history.txt")
        if hasattr(history, "history"):
            json.dump(history.history, open("history.json",'w'))
            np.savetxt("history.txt", history.history, delimiter=",")
        raise(e)

def update_model(state, weight):
    #state = [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    # K.tf.reset_default_graph()

    if weight == 'imagenet':
        model = ResNet50(state, include_top=True, weights='imagenet')
    elif weight == 'current':
        model = ResNet50(state, include_top=True, weights='current')
    # model.save_weights('current.h5')
    return model


def save_bottleneck_features(train_data_dir, val_data_dir, weights_path=WEIGHTS_PATH_NO_TOP, overwrite=False):
    bottleneck_features_name = "bottleneck_features.h5"
    if os.path.isfile(bottleneck_features_name):
        if overwrite:
            print("Overwriting bottleneck_features.h5")
            os.remove
        else:
            print("bottleneck_features.h5 exists, use overwrite=True to overwrite.")
            return
    return
    print(bottleneck_features_name + "is being created...~80GB for ImageNet200")
    batch_size=16
    img_width, img_height = 224, 224

    nb_train_samples = 100000
    nb_val_samples = 10000
    state = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    model = bottom_model(state, weights_path=weights_path)

    val_datagen = ImageDataGenerator(rescale=1. / 255)
    seed = 0
    val_generator = val_datagen.flow_from_directory(
        val_data_dir,
        target_size=(img_height, img_width),
        batch_size=batch_size,
        shuffle=True, seed=seed)


    train_datagen = ImageDataGenerator(rescale=1. / 255)

    train_generator = train_datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_height, img_width),
        batch_size=batch_size,
        shuffle=True, seed=seed)

    seed = 0

    np.random.seed(seed)
    train_index_array = np.random.permutation(100000)[:nb_train_samples]


    np.random.seed(seed)
    val_index_array = np.random.permutation(10000)[:nb_val_samples]

    chunk = 25
    train_parts = (nb_train_samples//batch_size)//chunk
    train_samples = nb_train_samples//train_parts
    # real_samples = samples - batch_size + samples%batch_size
    val_parts = (nb_val_samples//batch_size)//chunk
    val_samples = nb_val_samples//val_parts
    last = 0



    with h5py.File(bottleneck_features_name, 'w') as hf:

        train_labels = hf.create_dataset("train_labels",  data=np.take(train_generator.classes[:nb_train_samples], train_index_array))
        val_labels = hf.create_dataset("val_labels",  data=np.take(val_generator.classes[:nb_val_samples], val_index_array))

        val = hf.create_dataset("val",  (nb_val_samples, 14, 14, 1024), chunks=(64, 14, 14, 1024))

        for i in range(val_parts):
            print("Val done: " + str(100 * i/val_parts) + "%")
            max_q_size = 1
            val[i*val_samples:(i+1)*val_samples,:,:,:] = model.predict_generator(
                    val_generator, val_samples // batch_size, max_q_size=max_q_size)
            val_generator.batch_index -= max_q_size

        train = hf.create_dataset("training",  (nb_train_samples, 14, 14, 1024), chunks=(64, 14, 14, 1024))
        # dset[:,:,:,:] = model.predict_generator(
        #     train_generator, nb_train_samples // batch_size)
        for i in range(train_parts):
            print("Train done: " + str(100 * i/train_parts) + "%")
            max_q_size = 1
            train[i*train_samples:(i+1)*train_samples,:,:,:] = model.predict_generator(
                    train_generator, train_samples // batch_size, max_q_size=max_q_size)
            # recorrect for the over-calling of predict_generator by max_q_size
            train_generator.batch_index -= max_q_size

# EXAMPLE USAGE:
# train_data_dir = '/home/dev/Documents/10703-project/tiny-imagenet-200/train'
# val_data_dir = '/home/dev/Documents/10703-project/tiny-imagenet-200/val'
# save_bottleneck_features(train_data_dir=train_data_dir, val_data_dir=val_data_dir, overwrite=False)
# #
# state = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
# model = top_model(state, weights_path='top_half_weights.f5')
# finetune_top_model(model, save_weights='top_half_weights.f5')
