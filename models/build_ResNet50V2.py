"""
Hyperparameters.

SEED (int): A value that removes potential randomization in the script.
IMG_SIZE (int): The size of all images. Applies for both dimensions.
BATCH_SIZE (int): The size of subsets of images for the model before updating its weights for training.
EPOCHS (int): The amount of iterations the model does for training.
LEARNING_RATE (float): The speed that the model learns its weights from the error for training.
"""

SEED = 123
IMG_SIZE = 224
BATCH_SIZE = 64
EPOCHS = 50
LEARNING_RATE = 1e-3

"""
Imported libraries.

os:
!!! FINISH THIS !!!
"""

import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0" # TALK ABOUT WHY THIS CODE IS HERE

from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.layers import Rescaling
from tensorflow.keras.applications import ResNet50V2
from tensorflow.keras.optimizers import Adam

current_path = os.path.realpath(__file__)
parent_path = os.path.dirname(os.path.dirname(current_path))

train_path = os.path.join(parent_path, "data", "train")
valid_path = os.path.join(parent_path, "data", "valid")

# TALK ABOUT THE STRUCTURE OF THE DATA THAT THE CODE EXPECTS
train_data = image_dataset_from_directory(train_path, label_mode = "categorical", batch_size = BATCH_SIZE, image_size = (IMG_SIZE, IMG_SIZE), seed = SEED)
valid_data = image_dataset_from_directory(valid_path, label_mode = "categorical", batch_size = BATCH_SIZE, image_size = (IMG_SIZE, IMG_SIZE), seed = SEED)

normalization_layer = Rescaling(1. / 127.5, offset = -1)

normalized_train = train_data.map(lambda x, y: (normalization_layer(x), y))
normalized_valid = valid_data.map(lambda x, y: (normalization_layer(x), y))

model = ResNet50V2(
    include_top = True,
    weights = None,
    input_shape = (IMG_SIZE, IMG_SIZE, 3),
    classes = len(train_data.class_names)
)

model.compile(optimizer = Adam(learning_rate = LEARNING_RATE), loss = "categorical_crossentropy", metrics = ["accuracy"])

model.summary()

history = model.fit(normalized_train, epochs = EPOCHS, verbose = 1, validation_data = normalized_valid)