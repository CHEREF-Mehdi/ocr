from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K
import numpy as np 
import pandas as pd

train_data_dir = 'data/train'
validation_data_dir = 'data/validation'
test_data_dir='data/test'

num_classes = 10    

# input image dimensions 
img_width, img_height = 28, 28
epochs = 20
batch_size = 80


def make_model(dense_layer_sizes, filters, kernel_size, pool_size):
    
    model = Sequential()
    model.add(Conv2D(filters, kernel_size, padding='valid', input_shape=input_shape))
    model.add(Activation('relu'))
    
    model.add(Conv2D(filters, kernel_size, padding='valid'))
    model.add(Activation('relu'))    
    model.add(MaxPooling2D(pool_size=pool_size))
    model.add(Dropout(0.25))
    
    
    model.add(Flatten())    
    
    for layer_size in dense_layer_sizes:
        model.add(Dense(layer_size))
        model.add(Activation('relu'))
    
    model.add(Dropout(0.5))
    model.add(Dense(num_classes))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy',optimizer='nadam',metrics=['accuracy'])
    
    return model



if K.image_data_format() == 'channels_first':
    input_shape = (1, img_width, img_height)
else:
    input_shape = (img_width, img_height, 1)

datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    color_mode="grayscale",
    class_mode='categorical')

valid_generator = datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    color_mode="grayscale",
    class_mode='categorical')

test_generator = datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    color_mode="grayscale",
    class_mode='categorical')

model=make_model([64],28,(5,5),2)

STEP_SIZE_TRAIN=train_generator.n//train_generator.batch_size
STEP_SIZE_VALID=valid_generator.n//valid_generator.batch_size

model.fit_generator(
    train_generator,
    steps_per_epoch=STEP_SIZE_TRAIN,
    epochs=epochs,
    validation_data=valid_generator,
    validation_steps=STEP_SIZE_VALID)

print("Validation :")
print(model.evaluate_generator(generator=valid_generator,steps=STEP_SIZE_VALID))
print("Test :")
print(model.evaluate_generator(generator=test_generator,steps=STEP_SIZE_VALID))

STEP_SIZE_TEST=test_generator.n//test_generator.batch_size
test_generator.reset()
pred=model.predict_generator(test_generator,
steps=STEP_SIZE_TEST,
verbose=1)

predicted_class_indices=np.argmax(pred,axis=1)

labels = (train_generator.class_indices)
labels = dict((v,k) for k,v in labels.items())
predictions = np.asarray([labels[k] for k in predicted_class_indices])

pd.DataFrame(predictions).to_csv("results.csv", header=None, index=None)







