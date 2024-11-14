import tensorflow as tf
import keras
from symmetry_lens.group_convolution_layer import GroupConvolutionLayer
from symmetry_lens.group_convolution_layer2d import GroupConvolutionLayer2d

@tf.keras.utils.register_keras_serializable()
class EquivariantAdapter(keras.Model):
    def __init__(
        self,
        name="equivariant_adapter",
        *args,
        **kwargs
    ):
        super(EquivariantAdapter, self).__init__(name=name)
        self._args = args
        self._kwargs = kwargs

    def get_config(self):
        config = super(EquivariantAdapter, self).get_config()
        return config

    def build(self, input_shape=None):
        self._input_shape = input_shape
        self._group_convolution_layer = GroupConvolutionLayer(**self._kwargs)

    def call(self, x, lr_scaled_normalized_training_time=None, training=False):
        y = self._group_convolution_layer(
            x,
            lr_scaled_normalized_training_time,
            training=training,
        )
        return y
    
    @property
    def symmetry_generator(self):
        return self._group_convolution_layer._generator
    
    @property
    def group_convolution_matrix(self):
        return self._group_convolution_layer._lifting_map
    
@tf.keras.utils.register_keras_serializable()
class EquivariantAdapter2d(keras.Model):
    def __init__(
        self,
        name="equivariant_adapter_2d",
        *args,
        **kwargs
    ):
        super(EquivariantAdapter2d, self).__init__(name=name)
        self._args = args
        self._kwargs = kwargs

    def get_config(self):
        config = super(EquivariantAdapter2d, self).get_config()
        return config

    def build(self, input_shape=None):
        self._input_shape = input_shape
        self._group_convolution_layer = GroupConvolutionLayer2d(**self._kwargs)

    def call(self, x, lr_scaled_normalized_training_time=None, training=False):
        y = self._group_convolution_layer(
            x,
            lr_scaled_normalized_training_time,
            training=training,
        )
        return y
    
    @property
    def x_symmetry_generator(self):
        return self._group_convolution_layer._x_generator
        
    @property
    def y_symmetry_generator(self):
        return self._group_convolution_layer._y_generator
        
    @property
    def group_convolution_matrix(self):
        return self._group_convolution_layer._lifting_map