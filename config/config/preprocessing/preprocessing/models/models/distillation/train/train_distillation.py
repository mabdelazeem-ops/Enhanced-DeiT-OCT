import os
import json
import tensorflow as tf
from models.enhanced_deit import create_deit_student
from models.cnn_teacher import build_cnn_teacher
from distillation.sdma_loss import KnowledgeDistillationLoss

def main():
    # Load parameters
    with open("config/student_config.json", "r") as f:
        config = json.load(f)
        
    print("Building Teacher (ResNet-50) and Student (DeiT) architectures...")
    teacher = build_cnn_teacher(num_classes=config["num_classes"])
    student = create_deit_student(
        image_size=config["image_size"],
        num_classes=config["num_classes"],
        embed_dim=config["embed_dim"]
    )
    
    # Optimizer and Loss setup
    optimizer = tf.keras.optimizers.AdamW(
        learning_rate=config["learning_rate"], 
        weight_decay=config["weight_decay"]
    )
    kd_loss_fn = KnowledgeDistillationLoss(temperature=3.0, alpha=0.5)
    
    student.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.CategoricalCrossentropy(),
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc")]
    )
    
    print("Model pipeline compiled successfully. Ready for patient-stratified 5-fold cross-validation.")

if __name__ == "__main__":
    main()
