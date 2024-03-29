from tensorflow.keras.layers import Layer, Conv2D, UpSampling2D
from tensorflow.keras.models import Model
from configs import LEVEL_FEATURE_PYRAMID
import tensorflow as tf


class FeaturePyramid(Layer):
    """Builds the Feature Pyramid with the feature maps from the backbone.

    Attributes:
      num_classes: Number of classes in the dataset.
      backbone: The backbone to build the feature pyramid from.

    There are some modest changes for the FPN here. A pyramid is generated from P3 to P7.
    Some major changes are:
        P2 is not used now due to computational reasons.
        P6 is computed by strided convolution instead of downsampling.
        P7 is included additionally to improve the accuracy of large object detection.
    """

    def __init__(self, **kwargs):
        super(FeaturePyramid, self).__init__(name="FeaturePyramid", **kwargs)
        self.conv_c3_1x1 = Conv2D(256, 1, 1, "same")
        self.conv_c4_1x1 = Conv2D(256, 1, 1, "same")
        self.conv_c5_1x1 = Conv2D(256, 1, 1, "same")
        self.conv_c3_3x3 = Conv2D(256, 3, 1, "same")
        self.conv_c4_3x3 = Conv2D(256, 3, 1, "same")
        self.conv_c5_3x3 = Conv2D(256, 3, 1, "same")
        self.conv_c6_3x3 = Conv2D(256, 3, 2, "same")
        self.conv_c7_3x3 = Conv2D(256, 3, 2, "same")
        self.upsample_2x = UpSampling2D(2)

    def call(self, features, training=False):
        c3_output, c4_output, c5_output = features
        p3_output = self.conv_c3_1x1(c3_output)
        p4_output = self.conv_c4_1x1(c4_output)
        p5_output = self.conv_c5_1x1(c5_output)
        p4_output = p4_output + self.upsample_2x(p5_output)
        p3_output = p3_output + self.upsample_2x(p4_output)

        p3_output = self.conv_c3_3x3(p3_output)
        p4_output = self.conv_c4_3x3(p4_output)
        p5_output = self.conv_c5_3x3(p5_output)

        if LEVEL_FEATURE_PYRAMID[-1] == 5:
            return p3_output, p4_output, p5_output

        elif LEVEL_FEATURE_PYRAMID[-1] == 6:
            p6_output = self.conv_c6_3x3(c5_output)

            return p3_output, p4_output, p5_output, p6_output

        elif LEVEL_FEATURE_PYRAMID[-1] == 7:
            p6_output = self.conv_c6_3x3(c5_output)
            p7_output = self.conv_c7_3x3(tf.nn.relu(p6_output))

            return p3_output, p4_output, p5_output, p6_output, p7_output
