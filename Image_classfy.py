import tensorflow as tf
from tensorflow.keras import datasets, layers, models
from tensorflow.keras.models import load_model
from sklearn.metrics import confusion_matrix , classification_report
import matplotlib.pyplot as plt
import numpy as np
import glob
from tensorflow import keras
import cv2
import pandas as pd
from collections import Counter
import sklearn.preprocessing
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import MinMaxScaler
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
import numpy as np


# Mount to the google drive
drive.mount('/content/drive')


# Read image IDs into a dataframe
df = pd.read_csv('/content/drive/MyDrive/Train.csv')

# Create an array of Image Ids
imids = [imid for imid in df['Image_ID']]

# Path of Images
path = '/content/drive/MyDrive/Train_Images/*.jpg'
IMG_SIZE = 50


# The function reads images, scales them and stores them in a array, we also create an array of associated image classes 
def Create_image_classes(file):
    classes = []
    images = []

    for image in glob.glob(file):
        img = cv2.imread(image)
        scaled_img = cv2.resize(img, (IMG_SIZE,IMG_SIZE))
        images.append(scaled_img)
        strngs = image.split("/")
        Idm = strngs[-1]
        ind = imids.index(Idm.split(".")[0])
        classes.append(df['class'][ind])
    return images, classes


images, classes = Create_image_classes(path)


# The function reads images, scales them and stores them in a array, we also create an array of associated image classes 
def Create_image_classes(file):
    classes = []
    images = []

    for image in glob.glob(file):
        img = cv2.imread(image)
        scaled_img = cv2.resize(img, (IMG_SIZE,IMG_SIZE))
        images.append(scaled_img)
        strngs = image.split("/")
        Idm = strngs[-1]
        ind = imids.index(Idm.split(".")[0])
        classes.append(df['class'][ind])
    return images, classes


images, classes = Create_image_classes(path)

#Data Generator Object For Data Augmentation
data_augmentor = ImageDataGenerator(
    rotation_range = 90,
    zoom_range = 0.2,
    vertical_flip = True,
    shear_range = 0.2,
    horizontal_flip = True, 
    height_shift_range = 0.2,
    width_shift_range = 0.1,
    fill_mode='nearest'
)

# Function to perfome image augmentation 
def Augment_Images(images, classes):
    images_to_augment = list(glob.glob(path))
    
    for i in range(len(images_to_augment)):
        img = cv2.imread(images_to_augment[i])
        scaled_img = cv2.resize(img, (IMG_SIZE,IMG_SIZE))

        scaled_img = scaled_img.reshape((1,) + scaled_img.shape)
        aug = data_augmentor.flow(scaled_img)
        generated_images = [next(aug)[0].astype(np.uint8) for i in range(6)]
        
        for j in range(len(generated_images)):
            images.append(generated_images[j])
            classes.append(classes[i])

    return images, classes 



images, classes = Augment_Images(images, classes)

# Create one hot encoding for classes
y = LabelBinarizer().fit_transform(classes)

# Save images and classes as numpy array To just reload on the next Train
np.save('/content/drive/MyDrive/images.npy', images)
np.save('/content/drive/MyDrive/classes.npy', y)


train_data = np.load('/content/drive/MyDrive/images.npy')
labels = np.load('/content/drive/MyDrive/classes.npy')


# Split Data Into Test And Training Sets
x_train, x_test, y_train, y_test = train_test_split(
    train_data, labels, test_size=0.20, random_state=42)

# Scale The Data
x_train = x_train/ 255
x_test = x_test/ 255

tf.random.set_seed(0)

cnn = models.Sequential([
    
        layers.Conv2D(filters = 32, kernel_size = (1,1), activation = 'relu', input_shape = x_train.shape[1:]),
        layers.MaxPooling2D((2,2)),
    
        layers.Conv2D(filters = 64, kernel_size = (1,1), activation = 'relu'),
        layers.MaxPooling2D((2,2)),
        
        layers.Conv2D(filters = 64, kernel_size = (1,1), activation = 'relu'),
        layers.MaxPooling2D((2,2)),
        
        layers.Conv2D(filters = 64, kernel_size = (1,1), activation = 'relu'),
        layers.SpatialDropout2D(0.3),
        layers.MaxPooling2D((2,2)),
    
    
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.4),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.1),
        layers.Dense(3, activation='softmax')         
    ])

# Optimizer
optimizer = keras.optimizers.Adam(learning_rate = 1e-3 ) 

#Compile the Model
cnn.compile(optimizer = optimizer,
           loss = 'categorical_crossentropy',
           metrics = ['accuracy'])

# Train The Model
cnn.fit(x = x_train, y = y_train,
        batch_size = 32,
        epochs = 76,
        verbose = 1,
        validation_split=0.2,
        shuffle=True)

# Save The Model
cnn.save('/content/drive/MyDrive/zindi.h5')

# Reload the model from local directory
model = load_model('zindi.h5')

# Evaluate The Model
model.evaluate(x_test, y_test, verbose = 1)
