from typing import Dict, Tuple
import numpy as np
import tensorflow as tf


def create_new_layer(
    layer: tf.keras.layers.Conv2D, new_W: np.ndarray, new_b: np.ndarray
) -> tf.keras.layers.Conv2D:
    """
    This function creates the new layers
    """
    new_layer = tf.keras.layers.Conv2D(
        name=layer.name,
        filters=new_W.shape[-1],
        kernel_size=layer.get_config()["kernel_size"],
        strides=layer.get_config()["strides"],
        padding=layer.get_config()["padding"],
        data_format=layer.get_config()["data_format"],
        dilation_rate=layer.get_config()["dilation_rate"],
        activation=layer.get_config()["activation"],
        kernel_initializer=layer.get_config()["kernel_initializer"],
        kernel_regularizer=layer.get_config()["kernel_regularizer"],
        use_bias=True,
    )
    input_shape = (
        layer.input_shape[0],
        layer.input_shape[1],
        new_W.shape[-2],
    )
    new_layer.build(input_shape)
    new_layer.set_weights([new_W, new_b])
    return new_layer


def numpy_relu(x: np.ndarray) -> np.ndarray:
    return x * (x > 0)


def convert_channels_to_remove(
    channels_to_remove: np.ndarray, target_shape: tuple
) -> np.ndarray:
    """ """
    output = np.zeros(shape=target_shape)
    for elem in channels_to_remove:
        output[elem] = 1
    return output


def update_next_layer_magnitude(
    W_2: np.ndarray, b_2: np.ndarray, b_1: np.ndarray, removed_neurons: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    This function updates the consecutive layer buy removing the adequate input channels
    as welll as trasnfering the removed wieghts from the pruned layer in to the bias of
    the consecutive layer.
    """
    new_W2 = np.delete(arr=np.copy(W_2), obj=removed_neurons, axis=-2)
    channels_to_remove_ = convert_channels_to_remove(
        channels_to_remove=removed_neurons, target_shape=b_1.shape
    )
    bias = np.copy(b_1) * channels_to_remove_
    bias = numpy_relu(bias)
    bias = bias @ np.copy(np.sum(W_2, axis=(0, 1)))
    new_b2 = np.copy(b_2) + bias
    return new_W2, new_b2


def get_info_from_layer(
    layer: tf.keras.layers.Layer,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    This function finds the removed channels and returns the compact weights and biases
    """
    w = np.copy(layer.weights[0].numpy())
    b = np.copy(layer.weights[1].numpy())
    removed_neurons = list(np.where(np.sum(np.abs(w), axis=(0, 1, 2)) == 0)[0])
    new_w = np.delete(arr=np.copy(w), obj=removed_neurons, axis=-1)
    new_b = np.delete(arr=np.copy(b), obj=removed_neurons, axis=0)
    return new_w, new_b, removed_neurons


def get_info_from_layer_edited(
    w1: np.ndarray,
    b1: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    This function finds the removed channels and returns the compact weights and biases
    """
    w = np.copy(w1)
    b = np.copy(b1)
    removed_neurons = list(np.where(np.sum(np.abs(w), axis=(0, 1, 2)) == 0)[0])
    new_w = np.delete(arr=np.copy(w), obj=removed_neurons, axis=-1)
    new_b = np.delete(arr=np.copy(b), obj=removed_neurons, axis=0)
    return new_w, new_b, removed_neurons


def replace_layers_in_model(
    model: tf.keras.Model, new_layers: Dict[str, tf.keras.layers.Layer]
) -> tf.keras.Model:
    """
    This functio nedits the model with the new pruned layers
    """
    network_dict = {"input_layers_of": {}, "new_output_tensor_of": {}}
    for layer in model.layers:
        for node in layer._outbound_nodes:
            layer_name = node.outbound_layer.name
            if layer_name not in network_dict["input_layers_of"]:
                network_dict["input_layers_of"].update({layer_name: [layer.name]})
            else:
                if layer.name not in network_dict["input_layers_of"][layer_name]:
                    network_dict["input_layers_of"][layer_name].append(layer.name)
    network_dict["new_output_tensor_of"].update({model.layers[0].name: model.input})
    model_outputs = []
    for layer in model.layers[1:]:
        layer_input = [
            network_dict["new_output_tensor_of"][layer_aux]
            for layer_aux in network_dict["input_layers_of"][layer.name]
        ]
        if len(layer_input) == 1:
            layer_input = layer_input[0]
        if layer.name in list(new_layers.keys()):
            x = layer_input
            x = new_layers[layer.name](x)
        else:
            copied_layer = type(layer).from_config(layer.get_config())
            if isinstance(
                copied_layer, (tf.keras.layers.Dense, tf.keras.layers.Conv2D)
            ):
                copied_layer.build(layer.input_shape)
                copied_layer.set_weights([w.numpy() for w in layer.weights])
            x = copied_layer(layer_input)
        network_dict["new_output_tensor_of"].update({layer.name: x})
        if layer_name in model.output_names:
            model_outputs.append(x)
    return tf.keras.Model(inputs=model.inputs, outputs=model_outputs[-1])


def edit_model(model: tf.keras.Model, layers_to_merge: list) -> tf.keras.Model:
    """
    This function takes a model pruned with zeros in the weights
    and actually trims the model architecture
    """
    new_layers = {}
    new_weights = {}
    for l, (curr_layer_name, next_layer_name) in enumerate(layers_to_merge):
        current_layer = model.get_layer(curr_layer_name)
        next_layer = model.get_layer(next_layer_name)
        if curr_layer_name in new_layers:
            new_W1, new_b1, removed_neurons = get_info_from_layer_edited(
                new_weights[current_layer.name][0], new_weights[current_layer.name][1]
            )
        else:
            new_W1, new_b1, removed_neurons = get_info_from_layer(current_layer)
        new_layers[current_layer.name] = create_new_layer(
            layer=current_layer, new_W=new_W1, new_b=new_b1
        )
        new_W2, new_b2 = update_next_layer_magnitude(
            W_2=np.copy(next_layer.weights[0].numpy()),
            b_2=np.copy(next_layer.weights[1].numpy()),
            b_1=np.copy(current_layer.weights[1].numpy()),
            removed_neurons=removed_neurons,
        )
        new_layers[next_layer.name] = create_new_layer(
            layer=next_layer, new_W=new_W2, new_b=new_b2
        )
        new_weights[current_layer.name] = [new_W1, new_b1]
        new_weights[next_layer.name] = [new_W2, new_b2]
    new_model = replace_layers_in_model(model=model, new_layers=new_layers)
    new_model.build((32, 32, 3))
    for layer in new_model.layers:
        if layer.name in new_weights:
            layer.set_weights(new_weights[layer.name])
    return new_model
