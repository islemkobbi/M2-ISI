import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
from cifar_utils import evaluate_on_cifar10
import tensorflow as tf
from typing import List, Optional, Dict, Tuple
import numpy as np

model_name = "ResNet20"
# model_name = "vggbn"



def layer_is_foldeable(
    layer: tf.keras.layers.Layer, previous_layer: Optional[tf.keras.layers.Layer] = None
) -> bool:
    """
    Cheks if the layer can be folded
    """
    
    if isinstance(layer, tf.keras.layers.BatchNormalization ) and isinstance(previous_layer, tf.keras.layers.Dense ):
        return True
    
    return False



    


def get_previous_layer(layer, model):
    for cpt, l in enumerate(model.layers):
        if layer.name in [n.outbound_layer.name for n in l._outbound_nodes]:
            return cpt
    return None


def find_layers_to_fold(model: tf.keras.Model) -> List[Tuple[int, int]]:
    """
    This function returns the list of indices of layers to fold.
    """
    output_list = []
    for layer_cpt, layer in enumerate(model.layers):
        prev_cpt = get_previous_layer(layer, model)
        if layer_is_foldeable(
            layer=layer,
            previous_layer=model.layers[prev_cpt] if prev_cpt else None,
        ):
            output_list.append((layer_cpt, prev_cpt))
    return output_list


def compute_new_kernel(
    epsilon: float,
    gamma: np.ndarray,
    beta: np.ndarray,
    mu: np.ndarray,
    sigma: np.ndarray,
    W: np.ndarray,
    b: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    This function replaces the computation
        gamma ((Wx+b) - mu) / (sqrt(sigma + epsilon)) + beta
    par
        (gamma W / sqrt(sigma + epsilon)) x + (beta + gamma (b - mu) / sqrt(sigma + epsilon))
    """
    raise NotImplementedError("new kernel computation not implemented")


def edit_folded_layers(
    id_layers_to_fold: list, model: tf.keras.Model
) -> Tuple[Dict[str, tf.keras.layers.Layer], list]:
    """
    This functions returns two outputs. The first one is a dict of layers to replace (essentially convs with the new kernels and biases).
    The second is a list of layers to removes (essentially the names of the batchnormalization layers to fold).
    """
    layers_to_replace = {}
    layers_to_remove = []
    for i_curr, i_prev in id_layers_to_fold:
        layers_to_remove.append(model.layers[i_curr].name)
        config = model.layers[i_prev].get_config()
        config["use_bias"] = True
        new_layer = type(model.layers[i_prev]).from_config(config)
        new_layer.build(model.layers[i_prev].input_shape)
        new_layer.set_weights(
            compute_new_kernel(
                epsilon=model.layers[i_curr].epsilon,
                gamma=model.layers[i_curr].weights[0].numpy(),
                beta=model.layers[i_curr].weights[1].numpy(),
                mu=model.layers[i_curr].weights[2].numpy(),
                sigma=model.layers[i_curr].weights[3].numpy(),
                W=model.layers[i_prev].weights[0].numpy(),
                b=(
                    model.layers[i_prev].weights[1].numpy()
                    if model.layers[i_prev].get_config()["use_bias"]
                    else None
                ),
            )
        )
        layers_to_replace[model.layers[i_prev].name] = new_layer
    return layers_to_replace, layers_to_remove


def get_graph_as_dict(model: tf.keras.Model) -> Tuple[Dict[str, Dict[str, list]], bool]:
    """
    This function returns a dictionnary of the layers and their corresponding
    input layers. This serves the purpose of re-defining the graph with
    new layers.
    """
    network_dict = {"input": {}, "new_output": {}}
    for layer in model.layers:
        for node in layer._outbound_nodes:
            layer_name = node.outbound_layer.name
            if layer_name not in network_dict["input"]:
                network_dict["input"].update({layer_name: [layer.name]})
            else:
                if layer.name not in network_dict["input"][layer_name]:
                    network_dict["input"][layer_name].append(layer.name)
    network_dict["new_output"].update({model.layers[0].name: model.input})
    return network_dict, False


def replace_layers_in_model(
    model: tf.keras.Model,
    layers_to_replace: Dict[str, tf.keras.layers.Layer],
    layers_to_remove: list,
) -> tf.keras.Model:
    """
    This function replaces a set of layers by new versions in a model
    [WARNING] the current version doesn't edit the nodes of the graph
            i.e. the _outbound_nodes of the layers contain the old version and
            new version of the edited layers.
    This increases the memory foot-print of the resulting graph.
    """
    network_dict, is_transformer = get_graph_as_dict(model=model)
    model_outputs = []
    model_layers = model.layers[1:]
    model_outputnames = model.output_names
    for layer in model_layers:
        layer_input = [
            network_dict["new_output"][layer_aux]
            for layer_aux in network_dict["input"][layer.name]
        ]
        if len(layer_input) == 1:
            layer_input = layer_input[0]
        if len(network_dict["input"][layer.name]) == 1:
            if layer.name in layers_to_replace:
                x = layers_to_replace[layer.name](layer_input)
            elif layer.name in layers_to_remove:
                x = layer_input
            else:
                x = layer(layer_input)
        else:
            if isinstance(
                layer,
                (
                    tf.keras.layers.Multiply,
                    tf.keras.layers.Concatenate,
                    tf.keras.layers.Add,
                ),
            ):
                copied_layer = type(layer).from_config(layer.get_config())
                x = layer(
                    [
                        network_dict["new_output"][layer_aux]
                        for layer_aux in network_dict["input"][layer.name]
                    ]
                )
            elif layer.name in layers_to_replace:
                x = layers_to_replace[layer.name](
                    [
                        network_dict["new_output"][layer_aux]
                        for layer_aux in network_dict["input"][layer.name]
                    ]
                )
            elif layer.name in layers_to_remove:
                x = layer_input
            else:
                copied_layer = type(layer).from_config(layer.get_config())
                x = copied_layer(
                    network_dict["new_output"][network_dict["input"][layer.name][1]]
                )
        network_dict["new_output"].update({layer.name: x})
        if layer.name in model_outputnames:
            model_outputs.append(x)
    new_model_inputs = model.inputs
    if model.inputs is None:
        new_model_inputs = network_dict["new_output"][
            network_dict["input"][model.layers[0].name][0]
        ]
    new_model = tf.keras.Model(inputs=new_model_inputs, outputs=model_outputs[-1])
    new_model._name = model.name
    return new_model


def fold_model(model: tf.keras.Model) -> tf.keras.Model:
    """
    This function implements the naive batch-normalization layer folding.
    """
    id_layers_to_fold = find_layers_to_fold(model=model)
    layers_to_replace, layers_to_remove = edit_folded_layers(
        id_layers_to_fold=id_layers_to_fold, model=model
    )
    return replace_layers_in_model(
        model=model,
        layers_to_replace=layers_to_replace,
        layers_to_remove=layers_to_remove,
    )


if __name__ == "__main__":

    resnet20 = tf.keras.models.load_model(f"{model_name}.h5", compile=False)

    print(
        f"Initial model has \033[96m{np.sum([np.prod(var.shape) for var in (resnet20.trainable_variables + resnet20.non_trainable_variables)])}\033[00m parameters."
    )

    folded_resnet20 = fold_model(resnet20)

    print(
        f"Initial Accuracy of the model is \033[95m{evaluate_on_cifar10(resnet20):.3f}%\033[00m"
    )
    print("---")
    print(
        f"After folding model has \033[96m{np.sum([np.prod(var.shape) for var in (folded_resnet20.trainable_variables + folded_resnet20.non_trainable_variables)])}\033[00m parameters."
    )
    print(
        f"Accuracy after folding of the model is \033[95m{evaluate_on_cifar10(folded_resnet20):.3f}%\033[00m"
    )
    folded_resnet20.save(
        filepath=f"folded_{model_name.lower()}.h5",
        include_optimizer=False,
    )
