# Character_Recognition_using_Machine_Learning
<h1>Importing Libraries</h1>
<h3>
import os<br>
import numpy as np<br>
from tensorflow.keras.models import Sequential<br>
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout<br>
from tensorflow.keras.preprocessing.image import ImageDataGenerator<br>
from tensorflow.keras.optimizers import Adam</h3>
<li>
  <ol>os: This module provides functions to interact with the operating system.</ol>
<ol>numpy (np): A library for numerical computations and array manipulations.</ol>
<ol>Sequential: A linear stack of layers for building a neural network.</ol>
<ol>Conv2D, MaxPooling2D, Flatten, Dense, Dropout: Different types of layers used in Convolutional Neural Networks (CNNs).</ol>
<ol>ImageDataGenerator: Used to perform data augmentation, generating batches of tensor image data with real-time augmentation.</ol>
<ol>Adam: An optimization algorithm for training the model.</ol></li>
<h2>Defining Constants
</h2>
IMG_WIDTH, IMG_HEIGHT = 48, 48<br>
BATCH_SIZE = 32<br>
EPOCHS = 20<br>
NUM_CLASSES = 26  # Assuming 26 character classes (A-Z)<br>
<li>IMG_WIDTH, IMG_HEIGHT: Dimensions of the input images (48x48 pixels).</li>
<li>BATCH_SIZE: Number of samples processed before the model is updated.
</li>
<li>EPOCHS: Number of full passes through the training dataset.</li>
<li>NUM_CLASSES: The number of classes for classification, initially set to 26 (A-Z).</li>

<h1>Paths for Training and Testing Data</h1>
train_data_dir = '../data/training_data'<br>
test_data_dir = '../data/testing_data'<br>
<li>train_data_dir: Path to the directory containing training data.</li>
<li>test_data_dir: Path to the directory containing testing data.</li>

<h1>Data Augmentation for Training
</h1>
train_datagen = ImageDataGenerator(<br>
    rescale=1.0/255,<br>
    rotation_range=10,<br>
    width_shift_range=0.1,<br>
    height_shift_range=0.1,<br>
    shear_range=0.1,<br>
    zoom_range=0.1,<br>
    horizontal_flip=False,<br>
    fill_mode='nearest'<br>
)<br>
<li>ImageDataGenerator: Prepares the training data with augmentation (scaling, rotating, shifting, zooming, etc.).</li>
<li>rescale: Rescales pixel values from [0, 255] to [0, 1].</li>
<li>rotation_range: Rotates the images randomly by up to 10 degrees.</li>
<li>width_shift_range, height_shift_range: Randomly shifts the image horizontally and vertically by up to 10%.</li>
<li>shear_range: Shears the image randomly.</li>
<li>zoom_range: Zooms into the image randomly.</li>
<li>horizontal_flip: False here means no horizontal flipping.</li>
<li>fill_mode: Defines how to fill newly introduced pixels (e.g., from rotation).</li>

<h1>Data Preprocessing for Testing
</h1>
test_datagen = ImageDataGenerator(rescale=1.0/255)<br>
<li>rescale: Like the training data, pixel values are rescaled between [0, 1]. No augmentation is applied to test data.</li>

<h1>Load and Preprocess the Training Data
</h1>
train_generator = train_datagen.flow_from_directory(<br>
    train_data_dir,<br>
    target_size=(IMG_WIDTH, IMG_HEIGHT),<br>
    color_mode='grayscale',<br>
    batch_size=BATCH_SIZE,<br>
    class_mode='categorical'<br>
)<br>
<li>flow_from_directory: Loads the training images from the specified directory.</li>
<li>target_size: Resizes all images to 48x48.</li>
<li>color_mode: Loads images in grayscale.</li>
<li>batch_size: Number of images per batch.</li>
<li>class_mode: Specifies categorical classification (e.g., 26 classes).</li>
<h1>Load and Preprocess the Testing Data</h1>

test_generator = test_datagen.flow_from_directory(<br>
    test_data_dir,<br>
    target_size=(IMG_WIDTH, IMG_HEIGHT),<br>
    color_mode='grayscale',<br>
    batch_size=BATCH_SIZE,<br>
    class_mode='categorical'<br>
)<br>
<li>test_generator: Loads testing data similarly, without augmentation but with rescaling.</li>
<h1>Adjusting Number of Classes</h1>
NUM_CLASSES = 36  # Change this to the number of classes you have<br>
<li>NUM_CLASSES: Updated to 36, representing the new number of output classes (likely includes digits or special characters).</li>
<h1>Defining the CNN Model</h1>

model = Sequential()<br>
model.add(Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=(IMG_WIDTH, IMG_HEIGHT, 1)))<br>
model.add(MaxPooling2D(pool_size=(2, 2)))<br>
model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))<br>
model.add(MaxPooling2D(pool_size=(2, 2)))<br>
model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))<br>
model.add(MaxPooling2D(pool_size=(2, 2)))<br>
model.add(Flatten())<br>
model.add(Dense(256, activation='relu'))<br>
model.add(Dropout(0.5))<br>
model.add(Dense(NUM_CLASSES, activation='softmax'))<be>
<li>Sequential(): Initializes a linear model where layers are added sequentially.</li>
<li>Conv2D: A convolutional layer with 32 filters of size 3x3, followed by ReLU activation. Input shape is specified as (48, 48, 1) for grayscale images.</li>
<li>MaxPooling2D: Reduces spatial dimensions using a 2x2 filter.</li>
<li>Flatten: Converts the 2D feature maps into a 1D vector for input into fully connected layers.</li>
<li>Dense: Fully connected (FC) layer with 256 neurons, followed by another FC layer with NUM_CLASSES neurons and softmax activation for multi-class classification.</li>
<li>Dropout: Regularization to prevent overfitting by randomly setting 50% of inputs to zero during training.</li>

<h1>Compiling the Model</h1>
model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])<br>
<li>optimizer: Uses the Adam optimizer with a learning rate of 0.001.</li>
<li>loss: Categorical crossentropy is used as the loss function for multi-class classification.</li>
<li>metrics: The model will report accuracy during training.</li>

<h1>Training the Model</h1>
history = model.fit(<br>
    train_generator,<br>
    steps_per_epoch=train_generator.samples // BATCH_SIZE,<br>
    epochs=EPOCHS<br>
)<br>
<li>fit: Trains the model using the training data</li>
<li>steps_per_epoch: Number of batches per epoch,calculated as total samples divided by batch size.</li>
<li>epochs: The number of complete passes through the dataset.</li>
<h1>Evaluating the Model</h1>
score = model.evaluate(test_generator)<br>
print(f'Test Loss: {score[0]} / Test Accuracy: {score[1]}')<br>
<li>evaluate: Evaluates the model on the testing data. Reports test loss and accuracy.</li>
<h1>Saving the Model</h1>
model.save('character_recognition_model.h5')<br>
<li>save: Saves the trained model to a file for later use.</li>
<h1>Plotting Training Accuracy and Loss</h1>
import matplotlib.pyplot as plt<br>
plt.plot(history.history['accuracy'])<br>
plt.title('Training Accuracy')<br>
plt.ylabel('Accuracy')<br>
plt.xlabel('Epoch')<br>
plt.legend(['Train'], loc='upper left')<br>
plt.show()<br>
plt.plot(history.history['loss'])<br>
plt.title('Training Loss')<br>
plt.ylabel('Loss')<br>
plt.xlabel('Epoch')<br>
plt.legend(['Train'], loc='upper left')<br>
plt.show()<br>
<li>matplotlib.pyplot: Plots the training accuracy and loss over epochs</li>
<li>history.history['accuracy']: Contains the accuracy metrics for each epoch.</li>
<li>history.history['loss']: Contains the loss values for each epoch.</li>

---------------------------------------------------------------------------------------------
#app.py
<h1> Importing Libraries</h1>
from flask import Flask, render_template, request, redirect, url_for<br>
import os<br>
import cv2<br>
import numpy as np<br>
from tensorflow.keras.models import load_model<br>
<li>Flask: A lightweight Python web framework used to create the web application.</li>
<li>render_template: Renders HTML files for displaying web pages.</li>
<li>request: Handles file uploads and form data.</li>
<li>redirect: Redirects to different URLs.</li>
<li>url_for: Generates URLs for the app's functions.</li>
<li>os: Interacts with the operating system, like creating directories or handling file paths.</li>
<li>cv2: OpenCV library for image processing.</li>
<li>numpy (np): For handling arrays and numerical operations.</li>
<li>load_model: Loads the pre-trained Keras model used for character recognition.</li>

<h1>Initializing the Flask App and Loading the Model</h1>

app = Flask(__name__)<br>
# Load your pre-trained model (make sure the path is correct)<br>
model = load_model('models/character_recognition_model.h5')<br>
<li>app: Creates an instance of the Flask app.</li>
<li>model: Loads the pre-trained character recognition model from a .h5 file. This model is used for prediction later.</li>

<h1> Setting Up the Upload Folder</h1>
UPLOAD_FOLDER = os.path.join(os.getcwd(), 'uploads')<br>
if not os.path.exists(UPLOAD_FOLDER):<br>
    os.makedirs(UPLOAD_FOLDER)<br>
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER<br>

<li>UPLOAD_FOLDER: Specifies the path to the directory where uploaded files (images) will be stored.</li>
<li>os.makedirs: Creates the directory if it does not exist.</li>

<h1>Image Preprocessing Function
</h1>
def preprocess_image(image_path, target_size=(48, 48)):<br>
    image = cv2.imread(image_path)  # Read the image<br>
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Convert to grayscale<br>
    image = cv2.resize(image, target_size)  # Resize to the target size (48x48 pixels)<br>
    image = image / 255.0  # Normalize the pixel values to [0, 1]<br>
    image = np.expand_dims(image, axis=-1)  # Add a channel dimension for grayscale<br>
    image = np.expand_dims(image, axis=0)  # Add a batch dimension<br>
    return image<br>
<li>cv2.imread: Reads the image from the given file path.</li>
<li>cv2.cvtColor: Converts the image to grayscale, which simplifies the input for the model.</li>
<li>cv2.resize: Resizes the image to the target size of 48x48 pixels, as required by the model.</li>
<li>image / 255.0: Normalizes the image pixel values to a range between 0 and 1.</li>
<li>np.expand_dims: Adds extra dimensions
<li>First expand_dims adds a channel dimension (grayscale images have 1 channel).</li>
<li>Second expand_dims adds a batch dimension, as the model expects a batch of images for prediction.</li>
</li>
<h1>Home Route
</h1>
@app.route('/')<br>
def home():<br>
    return render_template('index.html')<br>
  <li>@app.route('/'): Defines the route for the home page (/), which is the default landing page.</li>
  <li>render_template('index.html'): Renders the index.html file, which should contain the form for image upload.</li>









