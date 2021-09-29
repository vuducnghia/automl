import tensorflow as tf
from utils.convert_box import swap_xy, convert_to_xywh
from configs import INPUT_SHAPE
from keras.applications.vgg16 import preprocess_input


def random_flip_horizontal(image, boxes):
    """Flips image and boxes horizontally with 50% chance

    Arguments:
      image: A 3-D tensor of shape `(height, width, channels)` representing an
        image.
      boxes: A tensor with shape `(num_boxes, 4)` representing bounding boxes,
        having normalized coordinates.

    Returns:
      Randomly flipped image and boxes
    """
    if tf.random.uniform(()) > 0.5:
        image = tf.image.flip_left_right(image)
        boxes = tf.stack(
            [1 - boxes[:, 2], boxes[:, 1], 1 - boxes[:, 0], boxes[:, 3]], axis=-1
        )
    return image, boxes


def preprocess_data(sample):
    """Applies preprocessing step to a single sample

    Arguments:
        :param sample: A dict representing a single training sample.
        :param resize_shape:
    Returns:
      image: Resized and padded image with random horizontal flipping applied.
      bbox: Bounding boxes with the shape `(num_objects, 4)` where each box is
        of the format `[x, y, width, height]`.
      class_id: An tensor representing the class id of the objects, having
        shape `(num_objects,)`.
    """
    image = sample["image"]
    bbox = swap_xy(sample["objects"]["bbox"])
    class_id = tf.cast(sample["objects"]["label"], dtype=tf.int32)

    image, bbox = random_flip_horizontal(image, bbox)
    image = tf.image.resize(image, INPUT_SHAPE[:2])

    bbox = tf.stack(
        [
            bbox[:, 0] * INPUT_SHAPE[1],
            bbox[:, 1] * INPUT_SHAPE[0],
            bbox[:, 2] * INPUT_SHAPE[1],
            bbox[:, 3] * INPUT_SHAPE[0],
        ],
        axis=-1,
    )
    print(bbox)
    bbox = convert_to_xywh(bbox)
    return image, bbox, class_id
