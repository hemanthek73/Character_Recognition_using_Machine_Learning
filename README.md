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
