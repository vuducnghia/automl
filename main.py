import tensorflow as tf
import tensorflow_datasets as tfds

from model.loss import Loss
from utils.preprocessing import preprocessData
from utils.encode_label import LabelEncoder
from model.model import ODHyperModel, ObjectDetectionNet
from keras_tuner import RandomSearch
from configs import NUM_CLASSES, BATCH_SIZE, EPOCHS, LEARNING_RATES, LEARNING_RATE_BOUNDARIES, EPOCHS_TUNER, MAX_TRIALS

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.compat.v1.Session(config=config)


def create_dataset():
    (train_dataset, val_dataset), dataset_info = tfds.load("coco/2017",
                                                           split=["train", "validation"],
                                                           with_info=True,
                                                           data_dir="data_coco")

    label_encoder = LabelEncoder()
    autotune = tf.data.AUTOTUNE
    for i in train_dataset.take(1):
        print(preprocessData(i))

    train_dataset = train_dataset.map(preprocessData, num_parallel_calls=autotune)

    train_dataset = train_dataset.shuffle(1 * BATCH_SIZE)
    train_dataset = train_dataset.padded_batch(BATCH_SIZE,
                                               padding_values=(0.0, 1e-8, -1),
                                               drop_remainder=True)
    # for i in train_dataset.take(1):
    #     print(i)
    #     label_encoder.encode_batch(i[0], i[1], i[2])
    train_dataset = train_dataset.map(label_encoder.encode_batch, num_parallel_calls=autotune)
    train_dataset = train_dataset.apply(tf.data.experimental.ignore_errors())
    train_dataset = train_dataset.prefetch(autotune)

    val_dataset = val_dataset.map(preprocessData, num_parallel_calls=autotune)
    val_dataset = val_dataset.padded_batch(BATCH_SIZE,
                                           padding_values=(0.0, 1e-8, -1),
                                           drop_remainder=True)
    val_dataset = val_dataset.map(label_encoder.encode_batch, num_parallel_calls=autotune)
    val_dataset = val_dataset.apply(tf.data.experimental.ignore_errors())
    val_dataset = val_dataset.prefetch(autotune)

    return train_dataset, val_dataset, dataset_info

    # train_steps_per_epoch = dataset_info.splits["train"].num_examples // BATCH_SIZE
    # val_steps_per_epoch = dataset_info.splits["validation"].num_examples // BATCH_SIZE


def loadDataset(records_path='tfrecords/'):
    train_dataset = tf.data.TFRecordDataset(tf.io.gfile.glob(records_path + '*.tfrecord'))
    label_encoder = LabelEncoder()
    autotune = tf.data.AUTOTUNE

    train_dataset = train_dataset.map(preprocessData, num_parallel_calls=autotune)
    train_dataset = train_dataset.shuffle(8 * BATCH_SIZE)
    train_dataset = train_dataset.padded_batch(BATCH_SIZE,
                                               padding_values=(0.0, 1e-8, -1),
                                               drop_remainder=True)
    train_dataset = train_dataset.map(label_encoder.encode_batch, num_parallel_calls=autotune)
    train_dataset = train_dataset.apply(tf.data.experimental.ignore_errors())
    train_dataset = train_dataset.prefetch(autotune)
    return train_dataset


def main():
    # train_dataset, val_dataset, dataset_info = create_dataset()
    # for i in train_dataset.take(1):
    #     print(i)
    # print(dataset_info.splits["train"].num_examples)
    # print(dataset_info.splits["validation"].num_examples)

    train_dataset = loadDataset()

    hypermodel = ODHyperModel(num_classes=80)
    tuner = RandomSearch(
        hypermodel,
        objective="val_loss",
        max_trials=MAX_TRIALS,
        executions_per_trial=1,
        overwrite=True,
        directory="my_dir",
        project_name="helloworld",
    )
    tuner.search_space_summary()
    tuner.search(
        train_dataset.take(1),
        epochs=EPOCHS_TUNER,
        validation_data=train_dataset.take(1),
    )

    best_model = tuner.get_best_models()[0]
    best_model.fit(train_dataset, validation_data=train_dataset.take(1), epochs=EPOCHS)
    # best_model.save("best_model")

    # model = ObjectDetectionNet(None, NUM_CLASSES)
    # learning_rate_fn = tf.optimizers.schedules.PiecewiseConstantDecay(
    #     boundaries=LEARNING_RATE_BOUNDARIES, values=LEARNING_RATES
    # )
    # optimizer = tf.optimizers.SGD(learning_rate=learning_rate_fn, momentum=0.9)
    # loss_fn = Loss(NUM_CLASSES)
    # model.compile(loss=loss_fn, optimizer=optimizer)
    # model.fit(train_dataset.take(20000), validation_data=val_dataset, epochs=EPOCHS)
    # model.save("my_model")


if __name__ == "__main__":
    main()
