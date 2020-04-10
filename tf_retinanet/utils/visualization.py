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

import cv2
import numpy as np

from .colors import label_color


def draw_box(image, box, color, thickness=2):
    """
    Draws a box on an image with a given color.

    Arguments:
        image:      The image to draw on.
        box:        A list of 4 elements (x1, y1, x2, y2).
        color:      The color of the box.
        thickness:  The thickness of the lines to draw a box with.
    """
    box_array = np.array(box).astype(int)
    cv2.rectangle(
        image=image,
        pt1=(box_array[0], box_array[1]),
        pt2=(box_array[2], box_array[3]),
        color=color,
        thickness=thickness,
        lineType=cv2.LINE_AA)


def draw_caption(image, box, caption):
    """
    Draws a caption above the box in an image.

    Args:
        image:      The image to draw on.
        box:        A list of 4 elements (x1, y1, x2, y2).
        caption:    String containing the text to draw.
    """
    box_array = np.array(box).astype(int)
    cv2.putText(
        img=image,
        text=caption,
        org=(box_array[0], box_array[1] - 10),
        fontFace=cv2.FONT_HERSHEY_PLAIN,
        fontScale=1,
        color=(0, 0, 0),
        thickness=2)

    cv2.putText(
        img=image,
        text=caption,
        org=(box_array[0], box_array[1] - 10),
        fontFace=cv2.FONT_HERSHEY_PLAIN,
        fontScale=1,
        color=(255, 255, 255),
        thickness=1)


def draw_boxes(image, boxes, color, thickness=2):
    """
    Draws boxes on an image with a given color.

    Args:
        image:      The image to draw on.
        boxes:      A [N, 4] matrix (x1, y1, x2, y2).
        color:      The color of the boxes.
        thickness:  The thickness of the lines to draw boxes with.
    """
    for box in boxes:
        draw_box(
            image=image,
            box=box,
            color=color,
            thickness=thickness)


def draw_detections(
        image,
        boxes,
        scores,
        labels,
        color=None,
        label_to_name=None,
        score_threshold=0.5):
    """
    Draws detections in an image.

    Args:
        image:              The image to draw on.
        boxes:              A [N, 4] matrix (x1, y1, x2, y2).
        scores:             A list of N classification scores.
        labels:             A list of N labels.
        color:              The color of the boxes. By default the color from
                                keras_retinanet.utils.colors.label_color will be used.
        label_to_name:      (optional) Functor for mapping a label to a name.
        score_threshold:    Threshold used for determining what detections to draw.
    """
    selection = np.where(scores > score_threshold)[0]

    for i in selection:
        col = color if color is not None else label_color(labels[i])
        draw_box(
            image=image,
            box=boxes[i, :],
            color=col)

        # draw labels
        caption = (label_to_name(labels[i]) if label_to_name else labels[i]) + \
                ': {0:.2f}'.format(scores[i])

        draw_caption(
            image=image,
            box=boxes[i, :],
            caption=caption)


def draw_annotations(
        image,
        annotations,
        color=(0, 255, 0),
        label_to_name=None):
    """
    Draws annotations in an image.

    Args:
        image:          The image to draw on.
        annotations:    A [N, 5] matrix (x1, y1, x2, y2, label) or dictionary containing bboxes
                            (shaped [N, 4]) and labels (shaped [N]).
        color:          The color of the boxes. By default the color from
                            keras_retinanet.utils.colors.label_color will be used.
        label_to_name:  (optional) Functor for mapping a label to a name.
    """
    if isinstance(annotations, np.ndarray):
        annotations = {'bboxes': annotations[:, :4], 'labels': annotations[:, 4]}

    assert 'bboxes' in annotations
    assert 'labels' in annotations
    assert annotations['bboxes'].shape[0] == annotations['labels'].shape[0]

    for i in range(annotations['bboxes'].shape[0]):
        label = annotations['labels'][i]
        col = color if color is not None else label_color(label)
        caption = '{}'.format(label_to_name(label) if label_to_name else label)

        draw_caption(
            image=image,
            box=annotations['bboxes'][i],
            caption=caption)

        draw_box(
            image=image,
            box=annotations['bboxes'][i],
            color=col)
