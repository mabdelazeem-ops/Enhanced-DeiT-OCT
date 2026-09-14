import tensorflow as tf
from tensorflow.keras import losses

class KnowledgeDistillationLoss(tf.keras.losses.Loss):
    def __init__(self, temperature=3.0, alpha=0.5, name="kd_loss"):
        super().__init__(name=name)
        self.temperature = temperature
        self.alpha = alpha
        self.ce_loss = losses.CategoricalCrossentropy()
        self.kl_loss = losses.KLDivergence()

    def call(self, y_true, teacher_logits, student_logits):
        # Standard Cross-Entropy Loss
        student_ce = self.ce_loss(y_true, tf.nn.softmax(student_logits))
        
        # Soft targets KL-Divergence
        teacher_soft = tf.nn.softmax(teacher_logits / self.temperature)
        student_soft = tf.nn.softmax(student_logits / self.temperature)
        kd_soft_loss = self.kl_loss(teacher_soft, student_soft) * (self.temperature ** 2)
        
        # Total Weighted Distillation Loss
        total_loss = (1.0 - self.alpha) * student_ce + self.alpha * kd_soft_loss
        return total_loss
