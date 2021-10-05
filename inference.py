import tensorflow as tf
import tensorflow_datasets as tfds
from utils.postprocessing import DecodePredictions
from configs import INPUT_SHAPE, NUM_CLASSES
import matplotlib.pyplot as plt
import numpy as np


def visualize_detections(image, pred_boxes, gt_boxes, classes, scores, figsize=(7, 7), linewidth=1, color=[0, 0, 1]):
    """Visualize Detections"""
    image = np.array(image, dtype=np.uint8)

    image_shape = image.shape
    print(image_shape)
    plt.figure(figsize=figsize)
    plt.axis("off")
    plt.imshow(image)
    ax = plt.gca()
    for box, _cls, score in zip(pred_boxes, classes, scores):
        text = "{}: {:.2f}".format(_cls, score)
        x1, y1, x2, y2 = box
        x1 *= image_shape[1] / INPUT_SHAPE[1]
        x2 *= image_shape[1] / INPUT_SHAPE[1]
        y1 *= image_shape[0] / INPUT_SHAPE[0]
        y2 *= image_shape[0] / INPUT_SHAPE[0]

        w, h = x2 - x1, y2 - y1

        patch = plt.Rectangle(
            [x1, y1], w, h, fill=False, edgecolor=[0, 1, 0], linewidth=linewidth
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

    for box in gt_boxes:
        y_min, x_min, y_max, x_max = box
        x_min *= image_shape[1]
        x_max *= image_shape[1]
        y_min *= image_shape[0]
        y_max *= image_shape[0]

        patch = plt.Rectangle(
            [x_min, y_min], x_max - x_min, y_max - y_min, fill=False, edgecolor=color, linewidth=linewidth
        )
        ax.add_patch(patch)
    plt.show()


def prepare_image(image):
    image = tf.image.resize(image, INPUT_SHAPE[:2])
    image /= 127.5
    image -= 1.
    return tf.expand_dims(image, axis=0)


def compute_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    iou = interArea / float(boxAArea + boxBArea - interArea)

    return iou


def calculateAP(metrics):
    AP = [0] * NUM_CLASSES
    for id, m in enumerate(metrics):
        try:
            AP[id] = m["tp"] / (m["tp"] + m["fp"])
        except ZeroDivisionError:
            AP[id] = 0
    mAP = sum(AP) / NUM_CLASSES
    return AP, mAP


if __name__ == "__main__":
    (val_dataset), dataset_info = tfds.load("coco/2017", split="validation", with_info=True, data_dir="data")
    int2str = dataset_info.features["objects"]["label"].int2str

    model = tf.keras.models.load_model("my_model")
    metrics = [{"tp": 0, "fp": 0} for i in range(0, NUM_CLASSES)]
    gts, dets = [], []
    for sample in val_dataset.take(1):
        image = tf.cast(sample["image"], dtype=tf.float32)
        image_shape = image.shape
        input_image = prepare_image(image)

        predictions = model.predict(input_image)

        detections = DecodePredictions(confidence_threshold=0.5).process(predictions)
        num_detections = detections.valid_detections[0]

        class_names = [
            int2str(int(x)) for x in detections.nmsed_classes[0][:num_detections]
        ]

        label_gts = sample["objects"]["label"]
        bbox_gts = sample["objects"]["bbox"]
        label_dets = detections.nmsed_classes[0][:num_detections]
        bbox_dets = detections.nmsed_boxes[0][:num_detections]
        # print("label_dets: ", label_dets)
        # print("label_gts: ", label_gts)
        label_gts_checked = [False] * len(label_gts)
        print(label_gts_checked)

        for i in range(num_detections):
            label_pred = int(label_dets[i])

            used = False
            for j in range(len(label_gts)):
                if label_pred == int(label_gts[j]):
                    y_min, x_min, y_max, x_max = bbox_gts[j]
                    x1, y1, x2, y2 = bbox_dets[i] / INPUT_SHAPE[1]

                    iou = compute_iou((x_min, y_min, x_min + x_max, y_min + y_max), (x1, y1, x2, y2))
                    print("iou: ", iou)
                    if iou >= 0.5:
                        metrics[label_pred]["tp"] += 1
                        used = True
                        label_gts_checked[j] = True
                        break
            if not used:
                metrics[label_pred]["fp"] += 1

        for id, label in enumerate(label_gts_checked):
            if not label:
                metrics[label_gts[id]]["fp"] += 1

    ap, map = calculateAP(metrics)
    print(ap)
    print(map)
    # visualize_detections(
    #     image,
    #     detections.nmsed_boxes[0][:num_detections],
    #     bbox_gts,
    #     class_names,
    #     detections.nmsed_scores[0][:num_detections],
    # )
