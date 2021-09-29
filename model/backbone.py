from tensorflow.keras.applications import ResNet50, DenseNet169, DenseNet121, MobileNetV2, InceptionV3
from tensorflow.keras import Model
from configs import BACKBONE_TUNER, INPUT_SHAPE


class BackBone():
    def __init__(self, hp=None):
        super(BackBone, self).__init__()
        if hp is None:
            self.backbone = "MobileNetV2"
        else:
            self.backbone = hp.Choice("backbone", BACKBONE_TUNER)

    def get(self, training=False):
        if self.backbone == "ResNet50":
            """
            Builds ResNet50 with pre-trained imagenet weights
            conv3_block4_out    (None, 28, 28, 512)
            conv4_block6_out    (None, 14, 14, 1024)
            conv5_block3_out    (None, 7, 7, 2048)
            """
            base_model = ResNet50(include_top=False, input_shape=INPUT_SHAPE)
            base_model.trainable = training
            c3_output, c4_output, c5_output = [
                base_model.get_layer(layer_name).output
                for layer_name in ["conv3_block4_out", "conv4_block6_out", "conv5_block3_out"]
            ]
            return Model(inputs=[base_model.inputs], outputs=[c3_output, c4_output, c5_output])
        elif self.backbone == "InceptionV3":
            pass
        elif self.backbone == "DenseNet169":
            """
            Builds DenseNet169 with pre-trained imagenet weights

            pool3_relu    (None, 28, 28, 512)
            pool4_relu    (None, 14, 14, 1280)
            relu    (None, 7, 7, 1664)
            """

            base_model = DenseNet169(include_top=False, input_shape=INPUT_SHAPE)
            base_model.trainable = training
            c3_output, c4_output, c5_output = [
                base_model.get_layer(layer_name).output
                for layer_name in ["pool3_relu", "pool4_relu", "relu"]
            ]
            return Model(inputs=[base_model.inputs], outputs=[c3_output, c4_output, c5_output])

        elif self.backbone == "DenseNet121":
            """
            Builds DenseNet169 with pre-trained imagenet weights

            pool3_relu    (None, 28, 28, 512)
            pool4_relu    (None, 14, 14, 1024)
            relu    (None, 7, 7, 1024)
            """

            base_model = DenseNet121(include_top=False, input_shape=INPUT_SHAPE)
            base_model.trainable = training
            c3_output, c4_output, c5_output = [
                base_model.get_layer(layer_name).output
                for layer_name in ["pool3_relu", "pool4_relu", "relu"]
            ]
            return Model(inputs=[base_model.inputs], outputs=[c3_output, c4_output, c5_output])

        elif self.backbone == "MobileNetV2":
            """
            Builds MobileNetV2 with pre-trained imagenet weights

            block_6_expand_relu    (None, 28, 28, 192)
            block_13_expand_relu    (None, 14, 14, 576)
            out_relu    (None, 7, 7, 1280)
            """

            base_model = MobileNetV2(include_top=False, input_shape=INPUT_SHAPE)
            base_model.trainable = training
            c3_output, c4_output, c5_output = [
                base_model.get_layer(layer_name).output
                for layer_name in ["block_6_expand_relu", "block_13_expand_relu", "out_relu"]
            ]
            return Model(inputs=base_model.inputs, outputs=[c3_output, c4_output, c5_output])
