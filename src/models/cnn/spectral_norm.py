import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Layer, InputSpec

class SpectralNormalization(Layer):
    """
    Implémentation de la normalisation spectrale pour les couches convolutionnelles.
    Aide à stabiliser l'entraînement en contraignant la norme spectrale des poids.
    
    Arguments:
        layer: La couche à normaliser
        power_iterations: Nombre d'itérations pour estimer la norme spectrale
    """
    def __init__(self, layer, power_iterations=1, **kwargs):
        self.power_iterations = power_iterations
        if not isinstance(layer, tf.keras.layers.Layer):
            raise ValueError("'layer' doit être une instance de keras.layers.Layer")
        super(SpectralNormalization, self).__init__(**kwargs)
        self.layer = layer
    
    def build(self, input_shape):
        self.layer.build(input_shape)
        self.w = self.layer.kernel
        self.w_shape = self.w.shape.as_list()
        
        # Initialiser en utilisant la puissance itérative
        self.u = self.add_weight(
            name='u',
            shape=(1, self.w_shape[-1]),
            initializer=tf.initializers.TruncatedNormal(stddev=0.02),
            trainable=False
        )
        super(SpectralNormalization, self).build()
    
    def call(self, inputs):
        self._update_weights()
        output = self.layer(inputs)
        return output
    
    def _update_weights(self):
        w_reshaped = tf.reshape(self.w, [-1, self.w_shape[-1]])
        
        # Puissance itérative pour l'approximation de la norme spectrale
        for _ in range(self.power_iterations):
            v = tf.matmul(self.u, w_reshaped, transpose_b=True)
            v = tf.nn.l2_normalize(v)
            u = tf.matmul(v, w_reshaped)
            u = tf.nn.l2_normalize(u)
        
        # Calculer la norme spectrale et normaliser
        sigma = tf.matmul(tf.matmul(v, w_reshaped), u, transpose_b=True)
        self.u.assign(u)
        self.layer.kernel.assign(self.w / sigma)
    
    def compute_output_shape(self, input_shape):
        return self.layer.compute_output_shape(input_shape)
    
    def get_config(self):
        config = super(SpectralNormalization, self).get_config()
        config.update({'power_iterations': self.power_iterations})
        config.update({'layer': self.layer})
        return config