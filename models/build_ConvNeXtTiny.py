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
EPOCHS = 5
LEARNING_RATE = 1e-3

"""
Imported libraries.

os:
!!! FINISH THIS !!!
"""

import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0" # TALK ABOUT WHY THIS CODE IS HERE

import matplotlib.pyplot as plt

from tensorflow.keras import Model, Input
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.applications import ConvNeXtTiny
from tensorflow.keras.layers import GlobalAveragePooling2D, LayerNormalization, Dense
from tensorflow.keras.optimizers import Adam

current_path = os.path.realpath(__file__)
parent_path = os.path.dirname(os.path.dirname(current_path))

train_path = os.path.join(parent_path, "data", "train")
valid_path = os.path.join(parent_path, "data", "valid")

# TALK ABOUT THE STRUCTURE OF THE DATA THAT THE CODE EXPECTS
train_data = image_dataset_from_directory(train_path, label_mode = "categorical", batch_size = BATCH_SIZE, image_size = (IMG_SIZE, IMG_SIZE), seed = SEED)
valid_data = image_dataset_from_directory(valid_path, label_mode = "categorical", batch_size = BATCH_SIZE, image_size = (IMG_SIZE, IMG_SIZE), seed = SEED)

base_model = ConvNeXtTiny(
    include_top = False,
    weights = "imagenet",
    input_shape = (IMG_SIZE, IMG_SIZE, 3)
)

base_model.trainable = False
base_model.summary()

inputs = Input(shape = (IMG_SIZE, IMG_SIZE, 3))
x = base_model(inputs, training = False)
x = GlobalAveragePooling2D()(x)
x = LayerNormalization()(x)
outputs = Dense(len(train_data.class_names), activation = "softmax")(x)
model = Model(inputs, outputs)

model.compile(optimizer = Adam(learning_rate = LEARNING_RATE), loss = "categorical_crossentropy", metrics = ["accuracy"])

history = model.fit(train_data, epochs = EPOCHS, verbose = 1, validation_data = valid_data)

checkpoint_path = os.path.join(parent_path, "checkpoints")
graph_path = os.path.join(parent_path, "training_graphs", "ConvNeXtTiny_graphs")

model.save(os.path.join(checkpoint_path, "ConvNeXtTiny_model.keras"))

acc_fig = plt.figure()

acc_ax = acc_fig.add_subplot()
acc_ax.plot(history.history["accuracy"])
acc_ax.plot(history.history["val_accuracy"])
acc_ax.set_title("ConvNeXtTiny Accuracy")
acc_ax.set_xlabel("EPOCHS")
acc_ax.set_ylabel("Accuracy")
acc_ax.legend(["Train", "Validation"], loc = "upper left")

acc_fig.savefig(os.path.join(graph_path, "ConvNeXtTiny_acc.png"))
plt.close(acc_fig)

loss_fig = plt.figure()

loss_ax = loss_fig.add_subplot()
loss_ax.plot(history.history["loss"])
loss_ax.plot(history.history["val_loss"])
loss_ax.set_title("ConvNeXtTiny Loss")
loss_ax.set_xlabel("EPOCHS")
loss_ax.set_ylabel("Loss")
loss_ax.legend(["Train", "Validation"], loc = "upper left")

loss_fig.savefig(os.path.join(graph_path, "ConvNeXtTiny_loss.png"))
plt.close(loss_fig)