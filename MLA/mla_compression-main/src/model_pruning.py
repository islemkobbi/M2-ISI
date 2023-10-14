import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
from cifar_utils import evaluate_on_cifar10, get_cifar10, fine_tune_on_cifar
from AutoPrune import edit_model
import tensorflow as tf
import numpy as np

# model_name = "resnet20"
model_name = "vggbn"

if model_name == "resnet20":
    layers_that_can_be_pruned = [
        "conv2d_1",
        "conv2d_3",
        "conv2d_5",
        "conv2d_7",
        "conv2d_10",
        "conv2d_12",
        "conv2d_14",
        "conv2d_17",
        "conv2d_19",
    ]
    layers_to_merge = [
        ["conv2d_1", "conv2d_2"],
        ["conv2d_3", "conv2d_4"],
        ["conv2d_5", "conv2d_6"],
        ["conv2d_7", "conv2d_8"],
        ["conv2d_10", "conv2d_11"],
        ["conv2d_12", "conv2d_13"],
        ["conv2d_14", "conv2d_15"],
        ["conv2d_17", "conv2d_18"],
        ["conv2d_19", "conv2d_20"],
    ]
else:
    layers_that_can_be_pruned = ["conv2d", "conv2d_1"]
    layers_to_merge = [["conv2d", "conv2d_1"], ["conv2d_1", "conv2d_2"]]


def magnitude_pruning(num_neurons_to_remove: int, W: np.ndarray) -> np.ndarray:
    """
    Magnitude-based prunign removes neurons based on
    the norm of their corresponding weights.
    """
    raise NotImplementedError("magnitude pruning criterion not implemented")


def compute_gradients(
    inputs: tf.Variable, model: tf.keras.Model, w: tf.Variable
) -> np.ndarray:
    raise NotImplementedError("weight gradients computation not implemented")


def gradient_based(
    num_neurons_to_remove: int,
    model: tf.keras.Model,
    cpt_layer: int,
    train_set: tf.data.Dataset,
) -> np.ndarray:
    """
    Gradient-based prunign removes neurons based on the norm of the gradients
    with respect to their corresponding weights.
    """
    images = next(train_set)[0]
    grads = compute_gradients(
        inputs=images, model=model, w=model.layers[cpt_layer].weights[0]
    )
    raise NotImplementedError("gradient pruning criterion not implemented")


def apply_pruning_on_a_layer(
    num_neurons_to_remove: int,
    model: tf.keras.Model,
    cpt_layer: int,
    criterion: str,
    train_set: tf.data.Dataset,
) -> np.ndarray:
    """
    Selects and removes a number (num_neurons_to_remove) of neurons to remove
    in the weight_tensor of the specified layer.
    """
    W = model.layers[cpt_layer].weights[0].numpy()
    if criterion == "magnitude":
        return magnitude_pruning(num_neurons_to_remove=num_neurons_to_remove, W=W)
    elif criterion == "gradient":
        return gradient_based(
            num_neurons_to_remove=num_neurons_to_remove,
            model=model,
            cpt_layer=cpt_layer,
            train_set=train_set,
        )
    return W


def apply_pruning_on_a_model(
    model: tf.keras.Model, pruning_target: float, criterion: str = "magnitude"
) -> tf.keras.Model:
    """
    for each prunable layer, we remove the percentage (in 0;1) pruning_target
    of neurons by setting weights to zero.
    """
    train_set, _ = get_cifar10(128)
    global layers_that_can_be_pruned
    for cpt_layer, layer in enumerate(model.layers):
        if layer.name in layers_that_can_be_pruned:
            layer.set_weights(
                [
                    apply_pruning_on_a_layer(
                        num_neurons_to_remove=int(
                            pruning_target * layer.weights[0].numpy().shape[-1]
                        ),
                        model=model,
                        cpt_layer=cpt_layer,
                        criterion=criterion,
                        train_set=train_set,
                    )
                ]
                + [e.numpy() for e in layer.weights[1:]]
            )
    return model


def count_non_zero_parameters(model: tf.keras.Model) -> int:
    return np.sum(
        [
            np.count_nonzero(var.numpy())
            for var in (model.trainable_variables + model.non_trainable_variables)
        ]
    )


def count_parameters(model: tf.keras.Model) -> int:
    return np.sum(
        [
            np.prod(var.shape)
            for var in (model.trainable_variables + model.non_trainable_variables)
        ]
    )


def pruning_experiment(model_path: str, criterion: str, model_name: str):
    resnet20 = tf.keras.models.load_model(model_path, compile=False)
    print(
        f"\rInitial model has \033[96m{count_parameters(resnet20)}\033[00m parameters"
        f" of which \033[93m{count_non_zero_parameters(resnet20)}\033[00m are non-zero."
    )
    print(
        f"\rInitial Accuracy of the model is \033[95m{evaluate_on_cifar10(resnet20):.3f}%\033[00m"
    )
    pruned_resnet20 = apply_pruning_on_a_model(
        resnet20, pruning_target=0.10, criterion=criterion
    )
    print("\r---")
    print(
        f"\rAfter pruning ({criterion}) model has \033[96m{count_parameters(pruned_resnet20)}\033[00m parameters"
        f" of which \033[93m{count_non_zero_parameters(pruned_resnet20)}\033[00m are non-zero."
    )
    print(
        f"\rAccuracy after pruning ({criterion}) of the model is \033[95m{evaluate_on_cifar10(pruned_resnet20):.3f}%\033[00m"
    )
    print("---")
    global layers_to_merge
    new_model = edit_model(model=pruned_resnet20, layers_to_merge=layers_to_merge)
    print(
        f"\rAccuracy after edition ({criterion}) of the model is \033[95m{evaluate_on_cifar10(new_model):.3f}%\033[00m"
    )
    print(
        f"\rEdited model has \033[96m{count_parameters(new_model)}\033[00m parameters"
        f" of which \033[93m{count_non_zero_parameters(new_model)}\033[00m are non-zero."
    )
    print("\r---")
    fine_tune_on_cifar(model=new_model)
    print(
        f"\rAccuracy after fine-tuning ({criterion}) of the model is \033[95m{evaluate_on_cifar10(new_model):.3f}%\033[00m"
    )
    print(
        f"\rfine-tuned model has \033[96m{count_parameters(new_model)}\033[00m parameters"
        f" of which \033[93m{count_non_zero_parameters(new_model)}\033[00m are non-zero."
    )
    new_model.save(f"pruned_{model_name}.h5", include_optimizer=False)


if __name__ == "__main__":
    pruning_experiment(f"folded_{model_name}.h5", "magnitude", model_name)
    print("\n---\n")
    pruning_experiment(f"folded_{model_name}.h5", "gradient", model_name)
