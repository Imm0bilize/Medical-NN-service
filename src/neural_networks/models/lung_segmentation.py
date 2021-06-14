from tensorflow.keras.layers import *
from tensorflow.keras.models import Model

from ..config import IMG_SIZE


class LungSegmentation:
    def __init__(self, path_to_weights, n_filter=16, dropout=0.2, use_upsampling=False, use_dropout=True, concat_axis=-1):
        self.path_to_weights = path_to_weights
        self.n_filter = n_filter
        self.dropout = dropout
        self.use_upsampling = use_upsampling
        self.use_dropout = use_dropout
        self.concat_axis = concat_axis

    def build_unet(self):
        params = dict(kernel_size=(3, 3), activation="relu",
                      padding="same", kernel_initializer="he_uniform")

        params_trans = dict(kernel_size=(2, 2), strides=(2, 2), padding="same")

        inputs = Input((IMG_SIZE, IMG_SIZE, 1))
        encodeA = Conv2D(filters=self.n_filter, **params)(inputs)
        encodeA = Conv2D(filters=self.n_filter, **params)(encodeA)
        poolA = MaxPooling2D(pool_size=(2, 2))(encodeA)

        encodeB = Conv2D(filters=self.n_filter * 2, **params)(poolA)
        encodeB = Conv2D(filters=self.n_filter * 2, **params)(encodeB)
        poolB = MaxPooling2D(pool_size=(2, 2))(encodeB)

        encodeC = Conv2D(filters=self.n_filter * 4, **params)(poolB)
        if self.use_dropout:
            encodeC = SpatialDropout2D(self.dropout)(encodeC)
        encodeC = Conv2D(filters=self.n_filter * 4, **params)(encodeC)
        poolC = MaxPooling2D(pool_size=(2, 2))(encodeC)

        encodeD = Conv2D(filters=self.n_filter * 8, **params)(poolC)
        if self.use_dropout:
            encodeD = SpatialDropout2D(self.dropout)(encodeD)
        encodeD = Conv2D(filters=self.n_filter * 8, **params)(encodeD)
        poolD = MaxPooling2D(pool_size=(2, 2))(encodeD)
        encodeE = Conv2D(filters=self.n_filter * 16, **params)(poolD)
        encodeE = Conv2D(filters=self.n_filter * 16, **params)(encodeE)

        if self.use_upsampling:
            up = UpSampling2D(size=(2, 2), interpolation='bilinear')(encodeE)
        else:
            up = Conv2DTranspose(filters=self.n_filter * 8, **params_trans)(encodeE)
        concatD = concatenate([up, encodeD], axis=self.concat_axis)
        decodeC = Conv2D(filters=self.n_filter * 8, **params)(concatD)
        decodeC = Conv2D(filters=self.n_filter * 8, **params)(decodeC)

        if self.use_upsampling:
            up = UpSampling2D(size=(2, 2), interpolation='bilinear')(decodeC)
        else:
            up = Conv2DTranspose(filters=self.n_filter * 4, **params_trans)(decodeC)
        concatC = concatenate([up, encodeC], axis=self.concat_axis)
        decodeB = Conv2D(filters=self.n_filter * 4, **params)(concatC)
        decodeB = Conv2D(filters=self.n_filter * 4, **params)(decodeB)

        if self.use_upsampling:
            up = UpSampling2D(size=(2, 2), interpolation='bilinear')(decodeB)
        else:
            up = Conv2DTranspose(filters=self.n_filter * 2, **params_trans)(decodeB)

        concatB = concatenate([up, encodeB], axis=self.concat_axis)
        decodeA = Conv2D(filters=self.n_filter * 2, **params)(concatB)
        decodeA = Conv2D(filters=self.n_filter * 2, **params)(decodeA)

        if self.use_upsampling:
            up = UpSampling2D(size=(2, 2), interpolation='bilinear')(decodeA)
        else:
            up = Conv2DTranspose(filters=self.n_filter, **params_trans)(decodeA)

        concatA = concatenate([up, encodeA], axis=self.concat_axis)
        convOut = Conv2D(filters=self.n_filter, **params)(concatA)
        convOut = Conv2D(filters=self.n_filter, **params)(convOut)
        prediction = Conv2D(filters=1, kernel_size=(1, 1), activation="sigmoid")(convOut)

        model = Model(inputs=[inputs], outputs=[prediction], name='UNET')

        model.load_weights(self.path_to_weights)
        return model
