import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf

tf.random.set_seed(1234)
from conv2d_quantized import QConv2D
from cifar_utils import evaluate_on_cifar10, get_cifar10


def fit_scaling_parameters(model: tf.keras.Model):
    trainset, _ = get_cifar10(128)
    for cpt, (images, _) in enumerate(trainset):
        print(f"\rfine-tuning model : {100*(cpt) / 100:.1f}%", end="")
        model(images, training=True)
        if cpt == 100:
            break


def quantize_a_model(
    model: tf.keras.Model, weight_bits: int, activation_bits: int
) -> tf.keras.Model:
    """
    This function replaces all layers in the original model by their quantized counter parts
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
    edited_layers = []
    for layer in model.layers[1:]:
        layer_input = [
            network_dict["new_output_tensor_of"][layer_aux]
            for layer_aux in network_dict["input_layers_of"][layer.name]
        ]
        if len(layer_input) == 1:
            layer_input = layer_input[0]
        if isinstance(layer, tf.keras.layers.Conv2D):
            x = layer_input
            new_layer = QConv2D(
                activation_bits=activation_bits,
                weight_bits=weight_bits,
                filters=layer.get_config()["filters"],
                kernel_size=layer.get_config()["kernel_size"],
                stride=layer.get_config()["strides"],
                padding=layer.get_config()["padding"],
                use_bias=layer.get_config()["use_bias"],
                activation=layer.get_config()["activation"],
                dilation_rate=layer.get_config()["dilation_rate"],
                weight_initializer=layer.get_config()["kernel_initializer"],
                bias_initializer=layer.get_config()["bias_initializer"],
                name=layer.get_config()["name"],
            )
            new_layer.build(layer.input_shape)
            edited_layers.append(layer.name)
            x = new_layer(x)
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
    new_model = tf.keras.Model(inputs=model.inputs, outputs=model_outputs[-1])
    new_model.build((32,32,3))
    for new_layer, layer in zip(new_model.layers, model.layers):
        if layer.name in edited_layers:
            new_layer.set_weights([w.numpy() for w in layer.weights]+[w.numpy() for w in new_layer.weights[2:]])
    return new_model


if __name__ == "__main__":
    activation_bits = 8
    weight_bits = 8
    resnet20 = tf.keras.models.load_model("folded_resnet20.h5", compile=False)
    print(
        f"\rAccuracy of the initial model is \033[95m{evaluate_on_cifar10(resnet20):.3f}%\033[00m"
    )
    quantized_resnet20 = quantize_a_model(
        model=resnet20, weight_bits=weight_bits, activation_bits=activation_bits
    )
    print(
        f"\rAccuracy of the quantized model is \033[95m{evaluate_on_cifar10(quantized_resnet20):.3f}%\033[00m"
    )
    fit_scaling_parameters(quantized_resnet20)
    print(
        f"\rAccuracy of the quantized model (scaled) is \033[95m{evaluate_on_cifar10(quantized_resnet20):.3f}%\033[00m"
    )
    quantized_resnet20.save("quantized_resnet20.h5", include_optimizer=False)
