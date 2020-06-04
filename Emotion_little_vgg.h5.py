#!/usr/bin/env python
# coding: utf-8

# In[1]:


import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D


# In[3]:


num_classes=5
img_rows,img_cols=48,48
batch_size=8


# In[4]:


train_data_dir=r'C:\Users\asd\Desktop\python file\starting python\face expression\train'
validation_data_dir=r'C:\Users\asd\Desktop\python file\starting python\face expression\Validation'


# In[5]:


train_datagen=ImageDataGenerator(rescale=1./255,rotation_range=30,shear_range=0.3,zoom_range=0.3,width_shift_range=0.4,height_shift_range=0.4,
                                 horizontal_flip=True,vertical_flip=True)
validation_datagen=ImageDataGenerator(rescale=1./255)


# In[6]:


train_generator=train_datagen.flow_from_directory(train_data_dir,color_mode='grayscale',target_size=(img_rows,img_cols),batch_size=batch_size, class_mode='categorical',shuffle=True)
validation_generator=validation_datagen.flow_from_directory(validation_data_dir,color_mode='grayscale',target_size=(img_rows,img_cols),batch_size=batch_size, class_mode='categorical',shuffle=True)


# In[7]:


model=Sequential()


# In[8]:


model.add(Conv2D(32, (3, 3),padding= 'same', kernel_initializer='he_normal', input_shape=(img_rows,img_cols,1)))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(Conv2D(32, (3, 3),padding= 'same', kernel_initializer='he_normal', input_shape=(img_rows,img_cols,1)))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.2))


# In[9]:


model.add(Conv2D(64, (3, 3),padding= 'same', kernel_initializer='he_normal'))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(Conv2D(64, (3, 3),padding= 'same', kernel_initializer='he_normal'))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.2))


# In[10]:


model.add(Conv2D(128, (3, 3),padding= 'same', kernel_initializer='he_normal'))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(Conv2D(128, (3, 3),padding= 'same', kernel_initializer='he_normal'))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.2))


# In[11]:


model.add(Conv2D(256, (3, 3),padding= 'same', kernel_initializer='he_normal'))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(Conv2D(256, (3, 3),padding= 'same', kernel_initializer='he_normal'))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.2))


# In[12]:


model.add(Flatten())
model.add(Dense(64,kernel_initializer="he_normal"))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))


# In[13]:


model.add(Dense(64,kernel_initializer="he_normal"))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))


# In[14]:


model.add(Dense(num_classes,kernel_initializer='he_normal'))
model.add(Activation('softmax'))


# In[15]:


print(model.summary())


# In[16]:


from keras.optimizers import RMSprop,SGD,Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau


# In[17]:


checkpoint = ModelCheckpoint('Emotion_little_vgg.h5',
                             monitor='val_loss',
                             mode='min',
                             save_best_only=True,
                             verbose=1)


# In[18]:


earlystop = EarlyStopping(monitor='val_loss',
                          min_delta=0,
                          patience=3,
                          verbose=1,
                          restore_best_weights=True
                          )


# In[19]:


reduce_lr = ReduceLROnPlateau(monitor='val_loss',
                              factor=0.2,
                              patience=3,
                              verbose=1,
                              min_delta=0.0001)


# In[20]:


callbacks = [earlystop,checkpoint,reduce_lr]

model.compile(loss='categorical_crossentropy',
              optimizer = Adam(lr=0.001),
              metrics=['accuracy'])


# In[23]:


nb_train_samples = 24176
nb_validation_samples = 3006
epochs=20


# In[24]:


history=model.fit_generator(
                train_generator,
                steps_per_epoch=nb_train_samples//batch_size,
                epochs=epochs,
                callbacks=callbacks,
                validation_data=validation_generator,
                validation_steps=nb_validation_samples//batch_size)


# In[ ]:




