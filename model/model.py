from tensorflow.keras import Model
from .feature_pyramid import FeaturePyramid
from .head import build_head
from .backbone import get_backbone
import tensorflow as tf
import numpy as np
import os
from model.backbone import get_backbone
from model.loss import Loss
from keras_tuner import HyperModel


class ObjectDetectionNet(Model):

    def __init__(self, hp, num_classes):
        super(ObjectDetectionNet, self).__init__(name="ObjectDetectionNet")
        self.num_classes = num_classes

        # self.backbone = get_backbone(hp)
        self.backbone = get_backbone(hp, "MobileNetV2")
        self.fpn = FeaturePyramid()

        prior_probability = tf.constant_initializer(-np.log((1 - 0.01) / 0.01))
        self.cls_head = build_head(9 * num_classes, prior_probability)
        self.box_head = build_head(9 * 4, "zeros")

    def call(self, image, training=False):
        features = self.backbone(image)
        features = self.fpn(features, training=training)
        N = tf.shape(image)[0]
        cls_outputs = []
        box_outputs = []
        for feature in features:
            box_outputs.append(tf.reshape(self.box_head(feature), [N, -1, 4]))
            cls_outputs.append(tf.reshape(self.cls_head(feature), [N, -1, self.num_classes]))
        cls_outputs = tf.concat(cls_outputs, axis=1)
        box_outputs = tf.concat(box_outputs, axis=1)

        return tf.concat([box_outputs, cls_outputs], axis=-1)


class ODHyperModel(HyperModel):
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.learning_rate_fn = self.setup_learning_rate()
        self.loss_fn = Loss(num_classes)
        self.optimizer = tf.optimizers.SGD(learning_rate=self.learning_rate_fn, momentum=0.9)

    def build(self, hp):
        model = ObjectDetectionNet(hp, self.num_classes)

        model.compile(loss=self.loss_fn, optimizer=self.optimizer)

        return model

    def setup_learning_rate(self):
        learning_rates = [2.5e-06, 0.000625, 0.00125, 0.0025, 0.00025, 2.5e-05]
        learning_rate_boundaries = [125, 250, 500, 240000, 360000]
        learning_rate_fn = tf.optimizers.schedules.PiecewiseConstantDecay(
            boundaries=learning_rate_boundaries, values=learning_rates
        )

        return learning_rate_fn

def setup_learning_rate():
    learning_rates = [2.5e-06, 0.000625, 0.00125, 0.0025, 0.00025, 2.5e-05]
    learning_rate_boundaries = [125, 250, 500, 240000, 360000]
    learning_rate_fn = tf.optimizers.schedules.PiecewiseConstantDecay(
        boundaries=learning_rate_boundaries, values=learning_rates
    )

    return learning_rate_fn
def setup_callback(model_dir="my_dir"):
    callbacks_list = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(model_dir, "weights" + "_epoch_{epoch}"),
            monitor="loss",
            save_best_only=False,
            save_weights_only=True,
            verbose=1,
        )
    ]

    return callbacks_list
