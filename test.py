import os
import zipfile

import tensorflow as tf
from tensorflow import keras
import tensorflow_datasets as tfds
from utils.preprocessing import preprocess_data
from utils.encode_label import LabelEncoder
from model.model import setup_callback, ODHyperModel, setup_learning_rate, ObjectDetectionNet
from model.loss import Loss
from keras_tuner import RandomSearch

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.compat.v1.Session(config=config)

url = "https://github.com/srihari-humbarwadi/datasets/releases/download/v0.1.0/data.zip"
filename = os.path.join(os.getcwd(), "data.zip")
keras.utils.get_file(filename, url)

with zipfile.ZipFile("data.zip", "r") as z_fp:
    z_fp.extractall("./")

(train_dataset, val_dataset), dataset_info = tfds.load(
    "coco/2017", split=["train", "validation"], with_info=True, data_dir="data"
)
batch_size = 4
epochs = 10

label_encoder = LabelEncoder()
autotune = tf.data.AUTOTUNE
train_dataset = train_dataset.map(preprocess_data, num_parallel_calls=autotune)
train_dataset = train_dataset.shuffle(8 * batch_size)
train_dataset = train_dataset.padded_batch(
    batch_size=batch_size, padding_values=(0.0, 1e-8, -1), drop_remainder=True
)
train_dataset = train_dataset.map(
    label_encoder.encode_batch, num_parallel_calls=autotune
)
train_dataset = train_dataset.apply(tf.data.experimental.ignore_errors())
train_dataset = train_dataset.prefetch(autotune)

val_dataset = val_dataset.map(preprocess_data, num_parallel_calls=autotune)
val_dataset = val_dataset.padded_batch(
    batch_size=1, padding_values=(0.0, 1e-8, -1), drop_remainder=True
)
val_dataset = val_dataset.map(label_encoder.encode_batch, num_parallel_calls=autotune)
val_dataset = val_dataset.apply(tf.data.experimental.ignore_errors())
val_dataset = val_dataset.prefetch(autotune)

steps_per_epoch = int(40670 / batch_size)
validation_steps = int(5000 / batch_size)
model = ObjectDetectionNet(None, 80)
learning_rate_fn = setup_learning_rate()
loss_fn = Loss(80)
optimizer = tf.optimizers.SGD(learning_rate=learning_rate_fn, momentum=0.9)
model.compile(loss=loss_fn, optimizer=optimizer)
model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=epochs,
    # callbacks=callbacks_list,
    verbose=1,
)
# tuner = RandomSearch(
#     build_model,
#     objective="val_loss",
#     max_trials=2,
#     executions_per_trial=1,
#     overwrite=True,
#     directory="my_dir",
#     project_name="helloworld",
# )
# tuner.search_space_summary()
# tuner.search(
#     train_dataset,
#     epochs=2,
#     validation_data=val_dataset,
#     steps_per_epoch=steps_per_epoch,
#     validation_steps=validation_steps,
#     # callbacks=setup_callback
# )
# best_model = tuner.get_best_models(num_models=1)[0]
# best_model.fit(train_dataset, )
# best_model.save("best_model")

