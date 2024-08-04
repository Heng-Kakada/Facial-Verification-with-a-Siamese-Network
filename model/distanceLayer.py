from keras import ops
from keras.layers import Layer

# Siamese L1 Distance class
class DistanceLayer(Layer):
    # Init method - inheritance
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    # Magic happens here - similarity calculation
    def call(self, input_embedding, validation_embedding):
        return ops.absolute((input_embedding[0] - validation_embedding[0]))