from typing import Any, Dict, Tuple, Type
import tensorflow as tf


def update_weight_scales(
    weights: tf.Variable,
    weight_scales: tf.Variable,
    weight_bits: int = 8,
) -> tf.Variable:
    """
    Updates the weight scales based on the current floating weight values
    """
    raise NotImplementedError("quantization weight scales update not implemented")


def update_input_scales(
    I: tf.Variable,
    input_scales: tf.Variable,
    activation_bits: int = 8,
    ema: float = 0.9,
) -> tf.Variable:
    """
    Updates the activation scales based on the current input/activation values.
    We also use this function for output scales
    """
    raise NotImplementedError("quantization input scales update not implemented")


def quantize_inputs(
    inputs: tf.Variable, scales: tf.Tensor, activation_bits: int
) -> tf.Variable:
    """
    x -> clip(round(x * s))
    """
    raise NotImplementedError("quantization of the inputs not implemented")


def quantize_weights(
    weights: tf.Variable, scales: tf.Tensor, weight_bits: int
) -> tf.Variable:
    """
    w -> clip(round(w * s))
    """
    raise NotImplementedError("quantization of the weights not implemented")


def dequantize_output(
    output: tf.Variable, input_scales: tf.Tensor, weight_scales: tf.Tensor
) -> tf.Variable:
    """
    o -> o / (s_I s_W)
    """
    raise NotImplementedError("dequantize not implemented")


@tf.keras.utils.register_keras_serializable()
class QConv2D(tf.keras.layers.Layer):
    def __init__(
        self,
        filters: int,
        kernel_size: Tuple[int, int],
        stride: Tuple[int, int, int, int] = (1, 1, 1, 1),
        activation_bits: int = 8,
        weight_bits: int = 8,
        padding: str = "SAME",
        use_bias: bool = True,
        activation: str = "linear",
        dilation_rate: Tuple[int, int] = (1, 1),
        weight_initializer: tf.keras.initializers.Initializer = "glorot_uniform",
        bias_initializer: tf.keras.initializers.Initializer = "zeros",
        trainable: bool = True,
        name: str = None,
        **kwargs: Any
    ):
        super().__init__(trainable, name, **kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.stride = stride
        self.activation_bits = activation_bits
        self.weight_bits = weight_bits
        self.padding = padding.upper()
        self.use_bias = use_bias
        self.activation = activation
        self.dilation_rate = dilation_rate
        self.weight_initializer = weight_initializer
        self.bias_initializer = bias_initializer
        self.trainable = trainable
        self.__name = name

    def build(self, input_shape):
        self.kernel = self.add_weight(
            name=self.name + "/kernel",
            shape=(
                self.kernel_size[0],
                self.kernel_size[1],
                input_shape[-1],
                self.filters,
            ),
            initializer=self.weight_initializer,
            trainable=True,
        )
        if self.use_bias:
            self.b = self.add_weight(
                name=self.name + "/bias",
                shape=(self.filters,),
                initializer=self.bias_initializer,
                trainable=True,
            )
        self.weight_scales = self.add_weight(
            name=self.name + "/kernel_scale",
            shape=(self.filters,),
            initializer="ones",
            trainable=False,
        )
        self.input_scales = self.add_weight(
            name=self.name + "/input_scale",
            shape=(1,),
            initializer="ones",
            trainable=False,
        )

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config["filters"] = self.filters
        config["kernel_size"] = self.kernel_size
        config["stride"] = self.stride
        config["activation_bits"] = self.activation_bits
        config["weight_bits"] = self.weight_bits
        config["padding"] = self.padding
        config["use_bias"] = self.use_bias
        config["activation"] = self.activation
        config["dilation_rate"] = self.dilation_rate
        config["weight_initializer"] = self.weight_initializer
        config["bias_initializer"] = self.bias_initializer
        config["trainable"] = self.trainable
        config["name"] = self.__name
        return config

    def get_operation_scales(
        self, training: bool, input: tf.Variable
    ) -> Tuple[tf.Variable, tf.Variable]:
        """
        gets and updates based on inference or training mode.
        """
        weight_scales_to_use = tf.cond(
            pred=training,
            true_fn=lambda: update_weight_scales(
                weights=self.kernel,
                weight_scales=self.weight_scales,
                weight_bits=self.weight_bits,
            ),
            false_fn=lambda: self.weight_scales,
        )
        input_scales_to_use = tf.cond(
            pred=training,
            true_fn=lambda: update_input_scales(
                I=input,
                input_scales=self.input_scales,
                activation_bits=self.activation_bits,
                ema=0.99,
            ),
            false_fn=lambda: self.input_scales,
        )
        return weight_scales_to_use, input_scales_to_use

    def call(
        self, inputs: tf.Variable, training: bool, *args: Any, **kwargs: Any
    ) -> tf.Variable:
        if training is not None:
            training = tf.cast(x=training, dtype=tf.bool)
        else:
            training = tf.cast(x=tf.keras.backend.learning_phase(), dtype=tf.bool)
        weight_scales, input_scales = self.get_operation_scales(
            training=training, input=inputs
        )
        input_quantized = quantize_inputs(
            inputs=inputs, scales=input_scales, activation_bits=self.activation_bits
        )
        weights_quantized = quantize_weights(
            weights=self.kernel, scales=weight_scales, weight_bits=self.weight_bits
        )
        output = tf.nn.conv2d(
            input=input_quantized,
            filters=weights_quantized,
            strides=self.stride,
            padding=self.padding,
        )
        output = dequantize_output(
            output=output, input_scales=input_scales, weight_scales=weight_scales
        )
        if self.use_bias:
            output = tf.nn.bias_add(output, self.b)
        if self.activation != "linear":
            if isinstance(self.activation, str):
                output = tf.keras.layers.Activation(self.activation)(output)
            else:
                output = self.activation(output)
        return output
