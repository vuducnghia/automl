from tensorflow.keras import Model
from .feature_pyramid import FeaturePyramid
from .head import build_head
import tensorflow as tf
import numpy as np
from model.backbone import BackBone
from model.loss import Loss
from keras_tuner import HyperModel
from configs import LEARNING_RATES, LEARNING_RATE_BOUNDARIES


class ObjectDetectionNet(Model):
    def __init__(self, hp, num_classes):
        super(ObjectDetectionNet, self).__init__(name="ObjectDetectionNet")
        self.num_classes = num_classes

        self.backbone = BackBone(hp).get()
        self.fpn = FeaturePyramid()

        prior_probability = tf.constant_initializer(-np.log((1 - 0.01) / 0.01))
        self.cls_head = build_head(9 * num_classes, prior_probability)
        self.box_head = build_head(9 * 4, "zeros")

    def call(self, image, training=False):
        features = self.backbone(image, training=training)
        features = self.fpn(features)
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
        self.learning_rate_fn = tf.optimizers.schedules.PiecewiseConstantDecay(
            boundaries=LEARNING_RATE_BOUNDARIES, values=LEARNING_RATES
        )
        self.loss_fn = Loss(num_classes)
        self.optimizer = tf.optimizers.SGD(learning_rate=self.learning_rate_fn, momentum=0.9)

    def build(self, hp):
        model = ObjectDetectionNet(hp, self.num_classes)

        model.compile(loss=self.loss_fn, optimizer=self.optimizer)

        return model

