from tensorflow.keras.layers import InputLayer, Conv2D, ReLU
import tensorflow as tf


def build_head(output_filters, bias_init):
    """Builds the class/box predictions head.

    Arguments:
      output_filters: Number of convolution filters in the final layer.
      bias_init: Bias Initializer for the final convolution layer.

    Returns:
      A keras sequential model representing either the classification
        or the box regression head depending on `output_filters`.
    """
    head = tf.keras.Sequential([InputLayer(input_shape=[None, None, 256])])
    kernel_init = tf.initializers.RandomNormal(0.0, 0.01)
    for _ in range(4):
        head.add(Conv2D(256, 3, padding="same", kernel_initializer=kernel_init))
        head.add(ReLU())
    head.add(Conv2D(output_filters, 3, 1, padding="same", kernel_initializer=kernel_init, bias_initializer=bias_init))
    return head
