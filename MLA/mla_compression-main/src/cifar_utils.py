from typing import Tuple
import numpy as np
import tensorflow as tf
import os


def get_data() -> Tuple[tf.Variable, tf.Variable, tf.Variable, tf.Variable]:
    """
    Collects (and evenutally downloads) the Cifar10 dataset.
    Then preprocesses the inputs and converts the labels
    to one hot format.
    """
    data_loaded = False
    if "data" in os.listdir(os.path.join(os.path.dirname(os.path.dirname(__file__)))):
        if "x_train.npy" in os.listdir(
            os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
        ):
            x_train = np.load(
                os.path.join(
                    os.path.dirname(os.path.dirname(__file__)),
                    "data/x_train.npy",
                )
            )
            y_train = np.load(
                os.path.join(
                    os.path.dirname(os.path.dirname(__file__)),
                    "data/y_train.npy",
                )
            )
            x_test = np.load(
                os.path.join(
                    os.path.dirname(os.path.dirname(__file__)),
                    "data/x_test.npy",
                )
            )
            y_test = np.load(
                os.path.join(
                    os.path.dirname(os.path.dirname(__file__)),
                    "data/y_test.npy",
                )
            )
            data_loaded = True
    if not data_loaded:
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
        if "data" not in os.listdir(
            os.path.join(os.path.dirname(os.path.dirname(__file__)))
        ):
            os.mkdir(os.path.join(os.path.dirname(os.path.dirname(__file__)), "data"))
        np.save(
            os.path.join(
                os.path.dirname(os.path.dirname(__file__)), "data/x_train.npy"
            ),
            x_train,
        )
        np.save(
            os.path.join(
                os.path.dirname(os.path.dirname(__file__)), "data/y_train.npy"
            ),
            y_train,
        )
        np.save(
            os.path.join(os.path.dirname(os.path.dirname(__file__)), "data/x_test.npy"),
            x_test,
        )
        np.save(
            os.path.join(os.path.dirname(os.path.dirname(__file__)), "data/y_test.npy"),
            y_test,
        )
    input_shape = x_train.shape[1:]
    x_train = x_train.astype("float32") / 255
    x_test = x_test.astype("float32") / 255
    x_train_mean = np.mean(x_train, axis=0)
    x_train -= x_train_mean
    x_test -= x_train_mean
    y_train = tf.keras.utils.to_categorical(y_train, 10)
    y_test = tf.keras.utils.to_categorical(y_test, 10)
    return x_train, y_train, x_test, y_test


def lr_schedule(epoch: int) -> float:
    lr = 1e-3
    if epoch > 180:
        lr *= 0.5e-3
    elif epoch > 160:
        lr *= 1e-3
    elif epoch > 120:
        lr *= 1e-2
    elif epoch > 80:
        lr *= 1e-1
    return lr


def create_training_dataset(x_train, y_train, batch_size):
    train_dataset = tf.keras.preprocessing.image.ImageDataGenerator(
        featurewise_center=False,
        samplewise_center=False,
        featurewise_std_normalization=False,
        samplewise_std_normalization=False,
        zca_whitening=False,
        rotation_range=0,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
        vertical_flip=False,
    )
    train_dataset.fit(x_train)
    train_dataset = train_dataset.flow(x_train, y_train, batch_size=batch_size)
    return train_dataset


def create_test_dataset(x_test, y_test, batch_size):
    test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    test_dataset = test_dataset.batch(batch_size)
    return test_dataset


def get_cifar10(batch_size: int) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
    x_train, y_train, x_test, y_test = get_data()
    return (
        create_training_dataset(x_train, y_train, batch_size),
        create_test_dataset(x_test, y_test, batch_size),
    )


def evaluate_on_cifar10(model: tf.keras.Model) -> float:
    _, test_dataset = get_cifar10(128)
    model.compile(tf.keras.optimizers.Adam(), metrics=["accuracy"])
    return 100 * model.evaluate(test_dataset, verbose=0)[-1]


def cross_entropy_batch(
    y_true: tf.Tensor, y_pred: tf.Tensor, label_smoothing: float = 0.0
) -> tf.Tensor:
    """ """
    cross_entropy = tf.keras.losses.categorical_crossentropy(
        y_true, y_pred, label_smoothing=label_smoothing
    )
    cross_entropy = tf.reduce_mean(cross_entropy)
    return cross_entropy


def fine_tune_on_cifar(model: tf.keras.Model):
    trainset, _ = get_cifar10(128)
    optimizer = tf.keras.optimizers.Adam(0.0001)
    for cpt, (images, lbls) in enumerate(trainset):
        print(f"\rfine-tuning model : {100*(cpt) / 1000:.1f}%", end="")
        with tf.GradientTape() as tape:
            prediction = model(images, training=True)
            ce = cross_entropy_batch(
                y_true=lbls, y_pred=prediction, label_smoothing=0.1
            )
            loss = ce
            gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        if cpt == 1000:
            break
