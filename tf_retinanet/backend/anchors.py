"""
Copyright 2017-2019 Fizyr (https://fizyr.com)

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import tensorflow as tf


def shift(image_shape, features_shape, stride, anchors):
    """
    Produce shifted anchors based on shape of the image, shape of the feature map and stride.

    Args
        image_shape:    Shape of the input image.
        features_shape: Shape of the feature map.
        stride:         Stride to shift the anchors with over the shape.
        anchors:        The anchors to apply at each location.
    """
    # Compute the offset of the anchors based on the image shape and the feature map shape.
    offset_x = tf.keras.backend.cast(
        x=(image_shape[1] - (features_shape[1] - 1) * stride),
        dtype=tf.keras.backend.floatx()
        ) / 2.0
    offset_y = tf.keras.backend.cast(
        x=(image_shape[0] - (features_shape[0] - 1) * stride),
        dtype=tf.keras.backend.floatx()
        ) / 2.0

    shift_x = tf.keras.backend.arange(
        start=0,
        stop=features_shape[1],
        dtype=tf.keras.backend.floatx()
        ) * stride + offset_x
    shift_y = tf.keras.backend.arange(
        start=0,
        stop=features_shape[0],
        dtype=tf.keras.backend.floatx()
        ) * stride + offset_y

    shift_x, shift_y = tf.meshgrid(shift_x, shift_y)
    shift_x = tf.keras.backend.reshape(shift_x, [-1])
    shift_y = tf.keras.backend.reshape(shift_y, [-1])

    shifts = tf.keras.backend.stack([
        shift_x,
        shift_y,
        shift_x,
        shift_y
    ], axis=0)

    shifts = tf.keras.backend.transpose(shifts)
    number_of_anchors = tf.keras.backend.shape(anchors)[0]

    # Number of base points = feat_h * feat_w.
    k = tf.keras.backend.shape(shifts)[0]

    shifted_anchors = tf.keras.backend.reshape(
        x=anchors,
        shape=[1, number_of_anchors, 4]
        ) + tf.keras.backend.cast(
            x=tf.keras.backend.reshape(
                x=shifts,
                shape=[k, 1, 4]),
            dtype=tf.keras.backend.floatx())

    shifted_anchors = tf.keras.backend.reshape(
        x=shifted_anchors,
        shape=[k * number_of_anchors, 4])

    return shifted_anchors


def bbox_transform_inv(boxes, deltas, mean=None, std=None):
    """
    Applies deltas (usually regression results) to boxes (usually anchors).
    Before applying the deltas to the boxes, the normalization that was previously applied
        (in the generator) has to be removed.
    The mean and std are the mean and std as applied in the generator. They are unnormalized
        in this function and then applied to the boxes.

    Args
        boxes : np.array of shape (B, N, 4), where B is the batch size, N the number of boxes
                    and 4 values for (x1, y1, x2, y2).
        deltas: np.array of same shape as boxes. These deltas (d_x1, d_y1, d_x2, d_y2) are
                    a factor of the width/height.
        mean  : The mean value used when computing deltas (defaults to [0, 0, 0, 0]).
        std   : The standard deviation used when computing deltas
                    (defaults to [0.2, 0.2, 0.2, 0.2]).

    Returns
        A np.array of the same shape as boxes, but with deltas applied to each box.
        The mean and std are used during training to normalize the regression values
            (networks love normalization).
    """
    if mean is None:
        mean = [0, 0, 0, 0]
    if std is None:
        std = [0.2, 0.2, 0.2, 0.2]

    width = boxes[:, :, 2] - boxes[:, :, 0]
    height = boxes[:, :, 3] - boxes[:, :, 1]

    x_1 = boxes[:, :, 0] + (deltas[:, :, 0] * std[0] + mean[0]) * width
    y_1 = boxes[:, :, 1] + (deltas[:, :, 1] * std[1] + mean[1]) * height
    x_2 = boxes[:, :, 2] + (deltas[:, :, 2] * std[2] + mean[2]) * width
    y_2 = boxes[:, :, 3] + (deltas[:, :, 3] * std[3] + mean[3]) * height

    pred_boxes = tf.keras.backend.stack(
        x=[x_1, y_1, x_2, y_2],
        axis=2)

    return pred_boxes
