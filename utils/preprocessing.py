import tensorflow as tf
from configs import INPUT_SHAPE
import urllib.request
from pycocotools.coco import COCO
import os


def getAnnotations(coco, imgId, width, height):
    annIds = coco.getAnnIds(imgIds=imgId)
    anns = coco.loadAnns(annIds)
    bboxes, catIds = [], []
    for ann in anns:
        try:
            catId = ann['category_id']
            bbox = [ann['bbox'][0] / width,
                    ann['bbox'][1] / height,
                    ann['bbox'][2] / width,
                    ann['bbox'][3] / height]
        except:
            continue
        if (not None in bbox) and (None != catId):
            catIds.append(catId)
            bboxes += bbox

    return len(anns), catIds, bboxes


def createTfRecordDataset(img_dir="data", annotations_file='test.json', records_path="tfrecords/", num_samples=4096):
    coco = COCO(annotations_file)
    imgIds = coco.getImgIds()
    n = len(imgIds)
    total_tfrecords = n // num_samples
    if n % num_samples:
        total_tfrecords += 1
    print("{} TFRecord files will be created".format(total_tfrecords))

    for i in range(0, total_tfrecords):
        examples = []
        start = i * num_samples
        end = start + num_samples
        imgids = imgIds[start:end]

        for img in coco.loadImgs(imgids):
            with open(str(img_dir) + str(img['id']) + ".jpg", 'rb') as f:
                image_string = f.read()

            objects, catIds, bboxes = getAnnotations(coco, img['id'], img['width'], img['height'])

            # Create a Features message using tf.train.Example.
            example = tf.train.Example(features=tf.train.Features(feature={
                'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_string])),
                'height': tf.train.Feature(int64_list=tf.train.Int64List(value=[img['height']])),
                'width': tf.train.Feature(int64_list=tf.train.Int64List(value=[img['width']])),
                'id': tf.train.Feature(int64_list=tf.train.Int64List(value=[img['id']])),
                # objects-Number of objects in the image
                'objects': tf.train.Feature(int64_list=tf.train.Int64List(value=[objects])),
                # Follwing features hold all the annotations data given for the image
                # category_ids-List of aannotation category ids
                'category_ids': tf.train.Feature(int64_list=tf.train.Int64List(value=catIds)),
                # bboxes flattened into 1D list
                'bboxes': tf.train.Feature(float_list=tf.train.FloatList(value=bboxes)),
            }))
            examples.append(example)

        with tf.io.TFRecordWriter(records_path + 'coco' + str(i) + '.tfrecord') as writer:
            for j in examples:
                writer.write(j.SerializeToString())
        examples.clear()
        print("file {} created".format(i))


def randomFlipHorizontal(image, boxes):
    """Flips image and boxes horizontally with 50% chance

    Arguments:
      image: A 3-D tensor of shape `(height, width, channels)` representing an
        image.
      boxes: A tensor with shape `(num_boxes, 4)` representing bounding boxes,
        having normalized coordinates.
        [x, y , width, height]


    Returns:
      Randomly flipped image and boxes
    """
    if tf.random.uniform(()) > 0.5:
        image = tf.image.flip_left_right(image)
        boxes = tf.stack([1 - boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]], axis=-1)
    return image, boxes


def preprocessData(sample):
    """

    :param sample:
    bboxes [x, y, width, height]
    :return:
    """
    sample = tf.io.parse_single_example(
        sample,
        features={
            'image': tf.io.FixedLenFeature([], tf.string),
            'height': tf.io.FixedLenFeature([], tf.int64),
            'width': tf.io.FixedLenFeature([], tf.int64),
            'objects': tf.io.FixedLenFeature([], tf.int64),
            'category_ids': tf.io.VarLenFeature(tf.int64),
            'bboxes': tf.io.VarLenFeature(tf.float32)
        })
    image = tf.image.decode_jpeg(sample['image'], channels=3)
    objects = sample['objects']
    bboxes = sample["bboxes"]
    bboxes = tf.sparse.to_dense(bboxes)
    bboxes = tf.reshape(bboxes, [objects, 4])
    class_id = tf.sparse.to_dense(sample["category_ids"])
    class_id = tf.cast(class_id, dtype=tf.int32)

    image, bboxes = randomFlipHorizontal(image, bboxes)
    image = tf.image.resize(image, INPUT_SHAPE[:2])

    bbox = tf.stack(
        [
            bboxes[:, 0] * INPUT_SHAPE[1],
            bboxes[:, 1] * INPUT_SHAPE[0],
            bboxes[:, 2] * INPUT_SHAPE[1],
            bboxes[:, 3] * INPUT_SHAPE[0],
        ],
        axis=-1,
    )

    return image, bbox, class_id
