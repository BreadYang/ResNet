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
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense
import time


img_width, img_height = 224, 224

train_data_dir = '/media/dev/Maxtor/linux/image_net/ILSVRC2012_img_val'
validation_data_dir = '/media/dev/Maxtor/linux/image_net/ILSVRC2012_img_val'
nb_train_samples = 50000
nb_validation_samples = 50000
epochs = 50
batch_size = 16


def tf_config_allow_growth_gpu():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    K.set_session(sess)

def init_compile_model():
    # build the ResNet50 network
    model = applications.ResNet50(include_top=True, weights='imagenet')
    print('Model loaded.')

    # set the first x layers (up to the last conv block)
    # to non-trainable (weights will not be updated)
    # for layer in model.layers[:25]:
    #     layer.trainable = False

    # compile the model with a SGD/momentum optimizer
    # and a very slow learning rate.
    model.compile(loss='binary_crossentropy',
                  optimizer=optimizers.SGD(lr=1e-4, momentum=0.9),
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
        print("Run-time on last batch:", runtime, "seconds")
        print("Avg Run-time on last batch:", sum(self.run_time)/len(self.run_time), "seconds")

# fine-tune the model

tf_config_allow_growth_gpu()
model = init_compile_model()
train_generator, validation_generator = init_generators()
runtime_callback = PrintRuntime()
model.fit_generator(
    train_generator,
    steps_per_epoch=nb_train_samples/batch_size,
    epochs=epochs,
    validation_data=validation_generator,
    nb_val_samples=nb_validation_samples, verbose=2, callbacks=[runtime_callback])
