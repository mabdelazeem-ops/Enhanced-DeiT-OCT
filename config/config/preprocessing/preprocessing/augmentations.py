import tensorflow as tf

def mixup(image1, label1, image2, label2, alpha=0.2):
    """Applies Mixup augmentation to a pair of images and labels."""
    beta_dist = tf.random.experimental.stateless_gamma(shape=[], alpha=[alpha, alpha], seed=[42, 0])
    lambda_val = beta_dist[0] / (beta_dist[0] + beta_dist[1])
    
    mixed_image = lambda_val * image1 + (1.0 - lambda_val) * image2
    mixed_label = lambda_val * label1 + (1.0 - lambda_val) * label2
    return mixed_image, mixed_label
