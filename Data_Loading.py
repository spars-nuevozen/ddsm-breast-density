
# coding: utf-8

# In[1]:


import keras
import os
import shutil
import pandas as pd


# In[2]:


data = pd.read_csv('mass_roi_train.csv')
data = data[['filename', 'breast_density']]
data.head()


# In[3]:


data['breast_density'].unique()


# In[11]:


train = 'mass_roi_train_images/'
test = 'mass_roi_test_images/'
'''for i in data.values:
    if i[1] == 3:
        shutil.copy(train+i[0], "/home/nuevozen/roi_dataset_breast_density/train/3/")
    elif i[1] == 2:
        shutil.copy(train+i[0], "/home/nuevozen/roi_dataset_breast_density/train/2/")
    elif i[1] == 1:
        shutil.copy(train+i[0], "/home/nuevozen/roi_dataset_breast_density/train/1/")
    elif i[1] == 4:
        shutil.copy(train+i[0], "/home/nuevozen/roi_dataset_breast_density/train/4/")


# In[4]:


data_test = pd.read_csv('mass_roi_test.csv')
data_test = data_test[['filename', 'breast_density']]
data_test.head()
print(data_test['breast_density'].unique())
for i in data_test.values:
    if i[1] == 3:
        shutil.copy(train+i[0], "/home/nuevozen/roi_dataset_breast_density/test/3/")
    elif i[1] == 2:
        shutil.copy(train+i[0], "/home/nuevozen/roi_dataset_breast_density/test/2/")
    elif i[1] == 1:
        shutil.copy(train+i[0], "/home/nuevozen/roi_dataset_breast_density/test/1/")
    elif i[1] == 4:
        shutil.copy(train+i[0], "/home/nuevozen/roi_dataset_breast_density/test/4/")'''


# In[18]:


from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing import image
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, GlobalAveragePooling2D
from keras.layers import Dense, Activation, Dropout, Flatten
from keras import backend as K
from keras.models import Model
from keras.layers import Input
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        zca_whitening = True)
 
test_datagen = ImageDataGenerator(rescale=1./255)
 
training_set = train_datagen.flow_from_directory(
        'train/',
        target_size=(350, 350),
        batch_size=32,
        class_mode='categorical')
 
test_set = test_datagen.flow_from_directory(
        'test/',
        target_size=(350, 350),
        batch_size=32,
        class_mode='categorical')


# In[22]:


input_tensor = Input(shape=(350, 350, 1)) 
base_model = InceptionV3(weights='imagenet', include_top=False)

# add a global spatial average pooling layer
x = base_model.output
x = GlobalAveragePooling2D()(x)
# let's add a fully-connected layer
x = Dense(512, activation='relu')(x)
# and a logistic layer -- let's say we have 200 classes
predictions = Dense(5, activation='softmax')(x)

# this is the model we will train
model = Model(inputs=base_model.input, outputs=predictions)

# first: train only the top layers (which were randomly initialized)
# i.e. freeze all convolutional InceptionV3 layers
for layer in base_model.layers:
    layer.trainable = True

# compile the model (should be done *after* setting layers to non-trainable)
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics = ['accuracy'])


# In[ ]:


from keras.callbacks import ModelCheckpoint
checkpointer = ModelCheckpoint(filepath='weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5', verbose=1, save_best_only=True)
callbacks_list = [checkpointer]
model.fit_generator(
        training_set,
        steps_per_epoch=32,
        epochs=50,
        validation_data=test_set,
        callbacks=callbacks_list)

