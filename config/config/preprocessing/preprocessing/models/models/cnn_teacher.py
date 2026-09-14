import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras import layers, models

def build_cnn_teacher(input_shape=(224, 224, 3), num_classes=4):
    """Loads and configures the pre-trained ResNet-50 Teacher Model."""
    base_model = ResNet50(weights="imagenet", include_top=False, input_shape=input_shape)
    base_model.trainable = False  # Freeze baseline weights during distillation

    x = layers.GlobalAveragePooling2D()(base_model.output)
    x = layers.Dense(256, activation="relu")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="teacher_output")(x)

    model = models.Model(inputs=base_model.input, outputs=outputs, name="ResNet50_Teacher")
    return model
