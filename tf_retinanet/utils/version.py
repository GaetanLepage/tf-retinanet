"""
Copyright 2017-2020 Fizyr (https://fizyr.com)

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

from __future__ import print_function

import sys
import tensorflow as tf

MINIMUM_TF_VERSION = 1, 14, 0
BLACKLISTED_TF_VERSIONS = [
    (2, 0, 0),  # Has a number of memory leaks and issues with eager execution.
    (2, 0, 1),  # Has a number of memory leaks and issues with eager execution.
]


def tf_version():
    """
    Get the Tensorflow version.

    Returns:
        tuple of (major, minor, patch).
    """
    return tuple(map(int, tf.version.VERSION.split('-')[0].split('.')))


def tf_version_ok(minimum_tf_version=MINIMUM_TF_VERSION, blacklisted=BLACKLISTED_TF_VERSIONS):
    """
    Check if the current Tensorflow version is higher than the minimum version.
    """
    return tf_version() >= minimum_tf_version and tf_version() not in blacklisted


def assert_tf_version(minimum_tf_version=MINIMUM_TF_VERSION, blacklisted=BLACKLISTED_TF_VERSIONS):
    """
    Assert that the Tensorflow version is up to date.
    """
    detected = tf.version.VERSION
    required = '.'.join(map(str, minimum_tf_version))
    assert tf_version_ok(minimum_tf_version, blacklisted),\
            "You are using tensorflow version {}. The minimum required version is {}"\
            " (blacklisted: {}).".format(detected, required, blacklisted)


def check_tf_version():
    """
    Check that the Tensorflow version is up to date.
    If it isn't, print an error message and exit the script.
    """
    try:
        assert_tf_version()
    except AssertionError as exception:
        print(exception, file=sys.stderr)
        sys.exit(1)
