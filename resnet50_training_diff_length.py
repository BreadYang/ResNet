'''
Training and testing data structure
```
data/
    train/
        cat1/
            cat1001.jpg
            cat1002.jpg
            ...
        cat2/
            cat2001.jpg
            cat2002.jpg
            ...
    validation/
        cat1/
            cat1001.jpg
            cat1002.jpg
            ...
        cat2/
            cat2001.jpg
            cat2002.jpg
            ...
```
'''
import tensorflow as tf
import keras.backend as K
import keras
from keras import applications
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import Sequential, Model, load_model
from keras.layers import Dropout, Flatten, Dense
import time
import build_CNN
import action_generator

img_width, img_height = 224, 224

train_data_dir = '/home/dev/Documents/10703-project/tiny-imagenet-200/train'
validation_data_dir = '/home/dev/Documents/10703-project/tiny-imagenet-200/val'
nb_train_samples = 100000
nb_validation_samples = 10000
epochs = 50
batch_size = 64


def tf_config_allow_growth_gpu():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    K.set_session(sess)


def pop_layer(model):
    if not model.outputs:
        raise Exception('Sequential model cannot be popped: model is empty.')

    model.layers.pop()
    if not model.layers:
        model.outputs = []
        model.inbound_nodes = []
        model.outbound_nodes = []
    else:
        model.layers[-1].outbound_nodes = []
        model.outputs = [model.layers[-1].output]
    model.built = False

def init_compile_model():
    # build the ResNet50 network
    initial_model = applications.ResNet50(include_top=True)
    # state = [1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0]
    state = [1, 1, 1, 1, 0, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0]
    initial_model = build_CNN.update_model(state, 'imagenet')
    pop_layer(initial_model)
    last = initial_model.layers[-1].output
    preds = Dense(200, activation='softmax', name="fc2")(last)
    model = Model(initial_model.input, preds)
    # model = load_model('my_model.h5')
    model.load_weights('my_model1.h5')
    print('Model loaded.')
    model.summary()
    # pop_layer(model)
    # set the first x layers (up to the last conv block)
    # to non-trainable (weights will not be updated)
    #for layer in model.layers[:25]:
        #layer.trainable = False

    # compile the model with a SGD/momentum optimizer
    # and a very slow learning rate.
    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizers.SGD(lr=1e-3, momentum=0.9),
                  metrics=['accuracy'])
    print('Model compiled.')
    return model

def init_generators():
    # prepare data augmentation configuration
    train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

    test_datagen = ImageDataGenerator(rescale=1. / 255)

    train_generator = train_datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_height, img_width),
        batch_size=batch_size)

    validation_generator = test_datagen.flow_from_directory(
        validation_data_dir,
        target_size=(img_height, img_width),
        batch_size=batch_size)
    print('Data generators created.')
    return train_generator, validation_generator


class PrintRuntime(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.run_time = []
        self.begin = 0
    def on_batch_begin(self, batch, logs={}):
        self.begin = time.time()
    def on_batch_end(self, batch, logs={}):
        end = time.time()
        runtime = end-self.begin
        self.run_time.append(runtime)
        if len(self.run_time)%100:
            # print("Run-time on last batch:", runtime, "seconds")
            print("Avg Run-time on last batch:", sum(self.run_time)/len(self.run_time), "seconds")

# fine-tune the model

tf_config_allow_growth_gpu()
model = init_compile_model()
train_generator, validation_generator = init_generators()
runtime_callback = PrintRuntime()
model.fit_generator(
    train_generator,
    steps_per_epoch=10000/64,
    epochs=30,
    validation_data=validation_generator,
    validation_steps=1000/64, verbose=2)

model.save('my_model2.h5')
