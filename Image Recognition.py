#!/usr/bin/env python
# coding: utf-8

# In[2]:


import os

# Get current working directory
current_directory = os.getcwd()
print("Current working directory:", current_directory)

# Change working directory
new_directory = "E:\Python"
os.chdir(new_directory)

# Check if the directory has been changed
updated_directory = os.getcwd()
print("Updated working directory:", updated_directory)


# In[3]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers




# In[4]:


IMG_SIZE = (256, 256)
VALID_SPLIT = 0.3  # Validation split ratio
BATCH_SIZE = 32  # Batch size for training
SEED = 42  
PATH = "E:\Python\Downloads\Food Classification"


# In[6]:


train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    PATH,
    validation_split=VALID_SPLIT,
    subset="training",
    seed=SEED,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="categorical"
)

# Get validation image with generator
val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    PATH,
    validation_split=VALID_SPLIT,
    subset="validation",
    seed=SEED,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="categorical"
)


# In[7]:


classes = train_ds.class_names


# **************************************************************************************

# In[8]:


import matplotlib.pyplot as plt

fig, axes = plt.subplots(3, 3, figsize=(10, 10))

for i, (images, labels) in enumerate(train_ds.take(9)):
    ax = axes[i//3, i%3]
    ax.imshow(images[i].numpy().astype("uint8"))
    ax.set_title(classes[np.argmax(labels[i])])
    ax.axis("off")

plt.tight_layout()
plt.show()


# In[9]:


data_augmentation = keras.Sequential([
    layers.experimental.preprocessing.RandomFlip("horizontal"),
    layers.experimental.preprocessing.RandomRotation(0.1),
])

augmented_train_ds = train_ds.map(lambda x, y: (data_augmentation(x, training=True), y))
augmented_train_ds = augmented_train_ds.prefetch(buffer_size=32)
val_ds = val_ds.prefetch(buffer_size=32)


# In[10]:


def construct_model(input_shape, num_classes):
    # Define the input layer with the given shape
    inputs = keras.Input(shape=input_shape)
    
    # Apply the data augmentation to the inputs
    x = data_augmentation(inputs)

    # Rescale the pixel values from 0-255 to 0-1
    x = layers.experimental.preprocessing.Rescaling(1.0 / 255)(x)
    
    # Define the entry block of the model
    x = layers.Conv2D(64, 3, strides=2, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    x = layers.Conv2D(64, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    # Save the current state of the model
    previous_block_activation = x

    # Add several blocks to the model
    for size in [128, 256, 512, 512]:
        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(size, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(size, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.MaxPooling2D(3, strides=2, padding="same")(x)

        # Project the residual
        residual = layers.Conv2D(size, 1, strides=2, padding="same")(previous_block_activation)
        x = layers.add([x, residual])  # Add the residual back
        previous_block_activation = x  # Set aside the next residual

    # Add the final layers
    x = layers.SeparableConv2D(1024, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    x = layers.GlobalAveragePooling2D()(x)
    
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    
    # Return the constructed model
    return keras.Model(inputs, outputs)
print(IMG_SIZE + (3,))
model = construct_model(input_shape=IMG_SIZE + (3,), num_classes=len(classes))


# In[11]:


model.summary()


# In[ ]:


EPOCHS = 100
CALLBACK = [
    keras.callbacks.ModelCheckpoint("filepath='model.{epoch:02d}-{val_loss:.2f}.h5'"),
    keras.callbacks.EarlyStopping(patience=15)
]

model.compile(
    optimizer='adam',
    loss="categorical_crossentropy",
    metrics=["categorical_accuracy"],
)

history = model.fit(
    augmented_train_ds,
    epochs=EPOCHS,
    callbacks=CALLBACK,
    validation_data=val_ds,
)



# In[ ]:


import seaborn as sns

# Create a figure and axes
fig, ax = plt.subplots(1, 2, figsize=(20, 8))

# Summarize history for accuracy
sns.lineplot(x=range(len(history.history['categorical_accuracy'])), y=history.history['categorical_accuracy'], ax=ax[0], label='train')
sns.lineplot(x=range(len(history.history['val_categorical_accuracy'])), y=history.history['val_categorical_accuracy'], ax=ax[0], label='val')
ax[0].set_title('Accuracy vs Epoch')
ax[0].set_xlabel('epoch')
ax[0].set_ylabel('accuracy')

# Summarize history for loss
sns.lineplot(x=range(len(history.history['loss'])), y=history.history['loss'], ax=ax[1], label='train')
sns.lineplot(x=range(len(history.history['val_loss'])), y=history.history['val_loss'], ax=ax[1], label='val')
ax[1].set_title('Loss vs Epoch')
ax[1].set_xlabel('epoch')
ax[1].set_ylabel('loss')

# Display plots
plt.show()


# In[ ]:




