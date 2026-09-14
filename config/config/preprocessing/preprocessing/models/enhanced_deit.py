import tensorflow as tf
from tensorflow.keras import layers, models

class PatchEncoder(layers.Layer):
    def __init__(self, num_patches, projection_dim):
        super().__init__()
        self.num_patches = num_patches
        self.projection = layers.Dense(units=projection_dim)
        self.position_embedding = layers.Embedding(
            input_dim=num_patches, output_dim=projection_dim
        )

    def call(self, patch):
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        encoded = self.projection(patch) + self.position_embedding(positions)
        return encoded

def create_deit_student(image_size=224, patch_size=16, num_classes=4, embed_dim=768):
    """Builds the Data-efficient Image Transformer (DeiT) student model."""
    inputs = layers.Input(shape=(image_size, image_size, 3))
    
    # Patch Extraction
    num_patches = (image_size // patch_size) ** 2
    patches = layers.Conv2D(filters=embed_dim, kernel_size=patch_size, strides=patch_size)(inputs)
    patches = layers.Reshape((num_patches, embed_dim))(patches)
    
    # Encoder
    encoded_patches = PatchEncoder(num_patches, embed_dim)(patches)
    
    # Transformer Blocks
    x = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
    for _ in range(6):
        # Multi-Head Attention
        attention_output = layers.MultiHeadAttention(num_heads=8, key_dim=embed_dim)(x, x)
        x1 = layers.Add()([attention_output, x])
        x2 = layers.LayerNormalization(epsilon=1e-6)(x1)
        # MLP Network
        mlp = layers.Dense(embed_dim * 2, activation=tf.nn.gelu)(x2)
        mlp = layers.Dense(embed_dim)(mlp)
        x = layers.Add()([mlp, x1])
        
    # Representation Representation Layer
    representation = layers.GlobalAveragePooling1D()(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="classification_head")(representation)
    
    model = models.Model(inputs=inputs, outputs=outputs, name="DeiT_Student")
    return model
