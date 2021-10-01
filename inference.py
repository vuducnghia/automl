import tensorflow as tf
import tensorflow_datasets as tfds
from utils.postprocessing import DecodePredictions
from configs import INPUT_SHAPE
import matplotlib.pyplot as plt
import numpy as np


def visualize_detections(image, boxes, classes, scores, figsize=(7, 7), linewidth=1, color=[0, 0, 1]):
    """Visualize Detections"""
    image = np.array(image, dtype=np.uint8)

    image_shape = image.shape
    print(image_shape)
    plt.figure(figsize=figsize)
    plt.axis("off")
    plt.imshow(image)
    ax = plt.gca()
    for box, _cls, score in zip(boxes, classes, scores):
        text = "{}: {:.2f}".format(_cls, score)
        x1, y1, x2, y2 = box
        x1 *= image_shape[1] / INPUT_SHAPE[1]
        x2 *= image_shape[1] / INPUT_SHAPE[1]
        y1 *= image_shape[0] / INPUT_SHAPE[0]
        y2 *= image_shape[0] / INPUT_SHAPE[0]

        w, h = x2 - x1, y2 - y1
        patch = plt.Rectangle(
            [x1, y1], w, h, fill=False, edgecolor=color, linewidth=linewidth
        )
        ax.add_patch(patch)
        ax.text(
            x1,
            y1,
            text,
            bbox={"facecolor": color, "alpha": 0.4},
            clip_box=ax.clipbox,
            clip_on=True,
        )
    plt.show()
    # plt.imsave("image", ax)


def prepare_image(image):
    image = tf.image.resize(image, INPUT_SHAPE[:2])
    image /= 127.5
    image -= 1.
    return tf.expand_dims(image, axis=0)


(val_dataset), dataset_info = tfds.load("coco/2017", split="validation", with_info=True, data_dir="data")
int2str = dataset_info.features["objects"]["label"].int2str

model = tf.keras.models.load_model("my_model")
for sample in val_dataset.take(20):
    image = tf.cast(sample["image"], dtype=tf.float32)
    input_image = prepare_image(image)

    predictions = model.predict(input_image)

    detections = DecodePredictions(confidence_threshold=0.5).process(predictions)
    num_detections = detections.valid_detections[0]
    class_names = [
        int2str(int(x)) for x in detections.nmsed_classes[0][:num_detections]
    ]
    visualize_detections(
        image,
        detections.nmsed_boxes[0][:num_detections],
        class_names,
        detections.nmsed_scores[0][:num_detections],
    )
