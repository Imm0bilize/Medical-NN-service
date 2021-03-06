import os
import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.regularizers import *
import numpy as np


class Yolo:
    def __init__(self):
        self.STRIDES = np.array([8, 16, 32])
        
        YOLO_ANCHORS = [[[10, 13], [16, 30], [33, 23]],
                        [[30, 61], [62, 45], [59, 119]],
                        [[116, 90], [156, 198], [373, 326]]]
        self.ANCHORS = (np.array(YOLO_ANCHORS).T / self.STRIDES).T

    def convolutional(self, input_layer, filters_shape, downsample=False, activate=True, bn=True):
        if downsample:
            input_layer = ZeroPadding2D(((1, 0), (1, 0)))(input_layer)
            padding = 'valid'
            strides = 2
        else:
            strides = 1
            padding = 'same'

        conv = Conv2D(filters=filters_shape[-1], kernel_size=filters_shape[0], strides=strides,
                      padding=padding, use_bias=not bn, kernel_regularizer=l2(0.0005),
                      kernel_initializer=tf.random_normal_initializer(stddev=0.01),
                      bias_initializer=tf.constant_initializer(0.))(input_layer)
        if bn:
            conv = BatchNormalization()(conv)
        if activate == True:
            conv = LeakyReLU(alpha=0.1)(conv)

        return conv

    def residual_block(self, input_layer, input_channel, filter_num1, filter_num2):
        short_cut = input_layer
        conv = self.convolutional(input_layer, filters_shape=(1, 1, input_channel, filter_num1))
        conv = self.convolutional(conv, filters_shape=(3, 3, filter_num1, filter_num2))

        residual_output = short_cut + conv
        return residual_output

    def upsample(self, input_layer):
        return tf.image.resize(input_layer, (input_layer.shape[1] * 2, input_layer.shape[2] * 2), method='nearest')

    def darknet53(self, input_data):
        input_data = self.convolutional(input_data, (3, 3, 3, 32))
        input_data = self.convolutional(input_data, (3, 3, 32, 64), downsample=True)

        for i in range(1):
            input_data = self.residual_block(input_data, 64, 32, 64)

        input_data = self.convolutional(input_data, (3, 3, 64, 128), downsample=True)

        for i in range(2):
            input_data = self.residual_block(input_data, 128, 64, 128)

        input_data = self.convolutional(input_data, (3, 3, 128, 256), downsample=True)

        for i in range(8):
            input_data = self.residual_block(input_data, 256, 128, 256)

        route_1 = input_data
        input_data = self.convolutional(input_data, (3, 3, 256, 512), downsample=True)

        for i in range(8):
            input_data = self.residual_block(input_data, 512, 256, 512)

        route_2 = input_data
        input_data = self.convolutional(input_data, (3, 3, 512, 1024), downsample=True)

        for i in range(4):
            input_data = self.residual_block(input_data, 1024, 512, 1024)

        return route_1, route_2, input_data

    def YOLOv3(self, input_layer, NUM_CLASS):
        # After the input layer enters the Darknet-53 network, we get three branches
        route_1, route_2, conv = self.darknet53(input_layer)
        # See the orange module (DBL) in the figure above, a total of 5 Subconvolution operation
        conv = self.convolutional(conv, (1, 1, 1024, 512))
        conv = self.convolutional(conv, (3, 3, 512, 1024))
        conv = self.convolutional(conv, (1, 1, 1024, 512))
        conv = self.convolutional(conv, (3, 3, 512, 1024))
        conv = self.convolutional(conv, (1, 1, 1024, 512))
        conv_lobj_branch = self.convolutional(conv, (3, 3, 512, 1024))

        # conv_lbbox is used to predict large-sized objects , Shape = [None, 13, 13, 255]
        conv_lbbox = self.convolutional(conv_lobj_branch, (1, 1, 1024, 3 * (NUM_CLASS + 5)), activate=False, bn=False)

        conv = self.convolutional(conv, (1, 1, 512, 256))
        # upsample here uses the nearest neighbor interpolation method, which has the advantage that the
        # upsampling process does not need to learn, thereby reducing the network parameter
        conv = self.upsample(conv)

        conv = tf.concat([conv, route_2], axis=-1)
        conv = self.convolutional(conv, (1, 1, 768, 256))
        conv = self.convolutional(conv, (3, 3, 256, 512))
        conv = self.convolutional(conv, (1, 1, 512, 256))
        conv = self.convolutional(conv, (3, 3, 256, 512))
        conv = self.convolutional(conv, (1, 1, 512, 256))
        conv_mobj_branch = self.convolutional(conv, (3, 3, 256, 512))

        # conv_mbbox is used to predict medium-sized objects, shape = [None, 26, 26, 255]
        conv_mbbox = self.convolutional(conv_mobj_branch, (1, 1, 512, 3 * (NUM_CLASS + 5)), activate=False, bn=False)

        conv = self.convolutional(conv, (1, 1, 256, 128))
        conv = self.upsample(conv)

        conv = tf.concat([conv, route_1], axis=-1)
        conv = self.convolutional(conv, (1, 1, 384, 128))
        conv = self.convolutional(conv, (3, 3, 128, 256))
        conv = self.convolutional(conv, (1, 1, 256, 128))
        conv = self.convolutional(conv, (3, 3, 128, 256))
        conv = self.convolutional(conv, (1, 1, 256, 128))
        conv_sobj_branch = self.convolutional(conv, (3, 3, 128, 256))

        # conv_sbbox is used to predict small size objects, shape = [None, 52, 52, 255]
        conv_sbbox = self.convolutional(conv_sobj_branch, (1, 1, 256, 3 * (NUM_CLASS + 5)), activate=False, bn=False)

        return [conv_sbbox, conv_mbbox, conv_lbbox]

    def decode(self, conv_output, NUM_CLASS, i=0):
        # where i = 0, 1 or 2 to correspond to the three grid scales
        conv_shape = tf.shape(conv_output)
        batch_size = conv_shape[0]
        output_size = conv_shape[1]

        conv_output = tf.reshape(conv_output, (batch_size, output_size, output_size, 3, 5 + NUM_CLASS))

        conv_raw_dxdy = conv_output[:, :, :, :, 0:2]  # offset of center position
        conv_raw_dwdh = conv_output[:, :, :, :, 2:4]  # Prediction box length and width offset
        conv_raw_conf = conv_output[:, :, :, :, 4:5]  # confidence of the prediction box
        conv_raw_prob = conv_output[:, :, :, :, 5:]  # category probability of the prediction box

        # next need Draw the grid. Where output_size is equal to 13, 26 or 52
        y = tf.range(output_size, dtype=tf.int32)
        y = tf.expand_dims(y, -1)
        y = tf.tile(y, [1, output_size])
        x = tf.range(output_size, dtype=tf.int32)
        x = tf.expand_dims(x, 0)
        x = tf.tile(x, [output_size, 1])

        xy_grid = tf.concat([x[:, :, tf.newaxis], y[:, :, tf.newaxis]], axis=-1)
        xy_grid = tf.tile(xy_grid[tf.newaxis, :, :, tf.newaxis, :], [batch_size, 1, 1, 3, 1])
        xy_grid = tf.cast(xy_grid, tf.float32)

        # Calculate the center position of the prediction box:
        pred_xy = (tf.sigmoid(conv_raw_dxdy) + xy_grid) * self.STRIDES[i]
        # Calculate the length and width of the prediction box:
        pred_wh = (tf.exp(conv_raw_dwdh) * self.ANCHORS[i]) * self.STRIDES[i]

        pred_xywh = tf.concat([pred_xy, pred_wh], axis=-1)
        pred_conf = tf.sigmoid(conv_raw_conf)  # object box calculates the predicted confidence
        pred_prob = tf.sigmoid(conv_raw_prob)  # calculating the predicted probability category box object

        # calculating the predicted probability category box object
        return tf.concat([pred_xywh, pred_conf, pred_prob], axis=-1)

    def build_model(self, input_size=512, channels=3, training=False):
        NUM_CLASS = 1
        input_layer = Input([input_size, input_size, channels])

        conv_tensors = self.YOLOv3(input_layer, NUM_CLASS)

        output_tensors = []
        for i, conv_tensor in enumerate(conv_tensors):
            pred_tensor = self.decode(conv_tensor, NUM_CLASS, i)
            if training:
                output_tensors.append(conv_tensor)
            output_tensors.append(pred_tensor)

        model = tf.keras.Model(input_layer, output_tensors)
        model.load_weights(os.path.join(os.path.dirname(__file__), 'weights', 'yolov3_custom.h5'))
        return model
