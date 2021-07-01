import os

import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model

# from ..config import IMG_SIZE


class DamageSegmentation:
    def conv_block(self, inputs, num_filters):
        x = Conv2D(num_filters, 3, padding="same")(inputs)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)

        x = Conv2D(num_filters, 3, padding="same")(x)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)
        return x

    def decoder_block(self, inputs, skip_features, num_filters):
        x = Conv2DTranspose(num_filters, (2, 2), strides=2, padding="same")(inputs)
        x = Concatenate()([x, skip_features])
        x = self.conv_block(x, num_filters)
        return x

    def build_model(self):
        inputs = Input((512, 512, 3))
        densenet = tf.keras.applications.DenseNet121(include_top=False, weights=None, input_tensor=inputs)
        s1 = densenet.get_layer("input_1").output
        s2 = densenet.get_layer("conv1/relu").output
        s3 = densenet.get_layer("pool2_relu").output
        s4 = densenet.get_layer("pool3_relu").output

        b1 = densenet.get_layer("pool4_relu").output

        d1 = self.decoder_block(b1, s4, 512)
        d2 = self.decoder_block(d1, s3, 256)
        d3 = self.decoder_block(d2, s2, 128)
        d4 = self.decoder_block(d3, s1, 64)

        outputs = Conv2D(1, 1, padding="same", activation="sigmoid")(d4)
        model = Model(inputs=inputs, outputs=outputs)
        model.load_weights(os.path.join(os.path.dirname(__file__), 'weights', 'damage_segment_weights.h5'))
        return model
