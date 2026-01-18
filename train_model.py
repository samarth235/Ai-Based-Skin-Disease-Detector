import tensorflow as tf
import numpy as np
from tensorflow.keras.applications.xception import Xception, preprocess_input
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers.legacy import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import tensorflow.keras.backend as K

# =========================
# CONFIG
# =========================
IMG_SIZE = 299
BATCH_SIZE = 32
EPOCHS = 18

TRAIN_DIR = "dataset/train"

# =========================
# FOCAL LOSS (fixes class confusion)
# =========================
def focal_loss(gamma=2., alpha=0.25):
    def loss(y_true, y_pred):
        y_pred = K.clip(y_pred, 1e-7, 1 - 1e-7)
        return -K.sum(alpha * y_true * K.pow(1 - y_pred, gamma) * K.log(y_pred), axis=1)
    return loss

# =========================
# DATA GENERATORS
# =========================
train_gen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.25,
    shear_range=0.2,
    horizontal_flip=True,
    fill_mode="nearest",
    validation_split=0.2
)

train_data = train_gen.flow_from_directory(
    TRAIN_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="training",
    shuffle=True
)

val_data = train_gen.flow_from_directory(
    TRAIN_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="validation",
    shuffle=False
)

# Save correct class index mapping
np.save("models/class_indices.npy", train_data.class_indices)
print("Class mapping saved:", train_data.class_indices)

# =========================
# BASE MODEL — XCEPTION
# =========================
base_model = Xception(
    weights="imagenet",
    include_top=False,
    input_shape=(IMG_SIZE, IMG_SIZE, 3)
)

base_model.trainable = False

# =========================
# CLASSIFIER HEAD
# =========================
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(512, activation="relu")(x)
x = Dropout(0.4)(x)
x = Dense(256, activation="relu")(x)
x = Dropout(0.3)(x)
output = Dense(train_data.num_classes, activation="softmax")(x)

model = Model(inputs=base_model.input, outputs=output)

# =========================
# COMPILE STAGE 1
# =========================
model.compile(
    optimizer=Adam(learning_rate=2e-5),
    loss=focal_loss(),
    metrics=["accuracy"]
)

callbacks = [
    EarlyStopping(patience=5, restore_best_weights=True),
    ReduceLROnPlateau(patience=3, factor=0.3)
]

# =========================
# TRAIN STAGE 1
# =========================
model.fit(
    train_data,
    validation_data=val_data,
    epochs=EPOCHS,
    callbacks=callbacks
)

# =========================
# FINE-TUNING STAGE
# =========================
print("🔁 Fine-tuning Xception...")

base_model.trainable = True
for layer in base_model.layers[:-150]:
    layer.trainable = False

model.compile(
    optimizer=Adam(learning_rate=1e-5),
    loss=focal_loss(),
    metrics=["accuracy"]
)

model.fit(
    train_data,
    validation_data=val_data,
    epochs=6,
    callbacks=callbacks
)

# =========================
# SAVE MODEL
# =========================
model.save("models/skin_model_xception.h5")
print("✅ Xception model saved as models/skin_model_xception.h5")
