﻿# Character_Recognition_using_Machine_Learning
<h1>Importing Libraries</h1>
import os
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
<li>os: This module provides functions to interact with the operating system.
numpy (np): A library for numerical computations and array manipulations.
Sequential: A linear stack of layers for building a neural network.
Conv2D, MaxPooling2D, Flatten, Dense, Dropout: Different types of layers used in Convolutional Neural Networks (CNNs).
ImageDataGenerator: Used to perform data augmentation, generating batches of tensor image data with real-time augmentation.
Adam: An optimization algorithm for training the model.</li>
