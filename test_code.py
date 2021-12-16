################################################################################
# Copyright (c) 2021 ContinualAI.                                              #
# Copyrights licensed under the MIT License.                                   #
# See the accompanying LICENSE file for terms.                                 #
#                                                                              #
# Date: 26-06-2020                                                             #
# Author(s): Gabriele Graffieti                                                #
# E-mail: contact@continualai.org                                              #
# Website: avalanche.continualai.org                                           #
################################################################################
from pathlib import Path
from typing import Optional, Sequence, Union, Any
import torch
from PIL.Image import Image
from torch import Tensor
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor, ToPILImage, Compose, Normalize, \
    RandomRotation
import numpy as np

from avalanche.benchmarks import NCScenario, nc_benchmark
from avalanche.benchmarks.classic.classic_benchmarks_utils import \
    check_vision_benchmark
from avalanche.benchmarks.datasets import default_dataset_location
from avalanche.benchmarks.utils import AvalancheDataset

_default_mnist_train_transform = Compose([
    ToTensor(),
    Normalize((0.1307,), (0.3081,))
])

_default_mnist_eval_transform = Compose([
    ToTensor(),
    Normalize((0.1307,), (0.3081,))
])

def RotatedMNIST(
        n_experiences: int,
        *,
        seed: Optional[int] = None,
        rotations_list: Optional[Sequence[int]] = None,
        train_transform: Optional[Any] = _default_mnist_train_transform,
        eval_transform: Optional[Any] = _default_mnist_eval_transform,
        dataset_root: Union[str, Path] = None) -> NCScenario:
    """
    Creates a Rotated MNIST benchmark.

    If the dataset is not present in the computer, this method will
    automatically download and store it.

    Random angles are used to rotate the MNIST images in ``n_experiences``
    different manners. This means that each experience is composed of all the
    original 10 MNIST classes, but each image is rotated in a different way.

    The benchmark instance returned by this method will have two fields,
    `train_stream` and `test_stream`, which can be iterated to obtain
    training and test :class:`Experience`. Each Experience contains the
    `dataset` and the associated task label.

    A progressive task label, starting from "0", is applied to each experience.

    The benchmark API is quite simple and is uniform across all benchmark
    generators. It is recommended to check the tutorial of the "benchmark" API,
    which contains usage examples ranging from "basic" to "advanced".

    :param n_experiences: The number of experiences (tasks) in the current
        benchmark. It indicates how many different rotations of the MNIST
        dataset have to be created.
        The value of this parameter should be a divisor of 10.
    :param seed: A valid int used to initialize the random number generator.
        Can be None.
    :param rotations_list: A list of rotations values in degrees (from -180 to
        180) used to define the rotations. The rotation specified in position
        0 of the list will be applied to the task 0, the rotation specified in
        position 1 will be applied to task 1 and so on.
        If None, value of ``seed`` will be used to define the rotations.
        If non-None, ``seed`` parameter will be ignored.
        Defaults to None.
    :param train_transform: The transformation to apply to the training data
        after the random rotation, e.g. a random crop, a normalization or a
        concatenation of different transformations (see torchvision.transform
        documentation for a comprehensive list of possible transformations).
        If no transformation is passed, the default train transformation
        will be used.
    :param eval_transform: The transformation to apply to the test data
        after the random rotation, e.g. a random crop, a normalization or a
        concatenation of different transformations (see torchvision.transform
        documentation for a comprehensive list of possible transformations).
        If no transformation is passed, the default test transformation
        will be used.
    :param dataset_root: The root path of the dataset. Defaults to None, which
        means that the default location for 'mnist' will be used.

    :returns: A properly initialized :class:`NCScenario` instance.
    """

    if rotations_list is not None and len(rotations_list) != n_experiences:
        raise ValueError("The number of rotations should match the number"
                         " of incremental experiences.")

    if rotations_list is not None and any(180 < rotations_list[i] < -180
                                          for i in range(len(rotations_list))):
        raise ValueError("The value of a rotation should be between -180"
                         " and 180 degrees.")

    list_train_dataset = []
    list_test_dataset = []
    rng_rotate = np.random.RandomState(seed)
    import pdb;pdb.set_trace()
    mnist_train, mnist_test = _get_mnist_dataset(dataset_root)

    # for every incremental experience
    for exp in range(n_experiences):
        if rotations_list is not None:
            rotation_angle = rotations_list[exp]
        else:
            # choose a random rotation of the pixels in the image
            rotation_angle = rng_rotate.randint(-180, 181)

        rotation = RandomRotation(degrees=(rotation_angle, rotation_angle))

        rotation_transforms = dict(
            train=(rotation, None),
            eval=(rotation, None)
        )

        # Freeze the rotation
        rotated_train = AvalancheDataset(
            mnist_train,
            transform_groups=rotation_transforms,
            initial_transform_group='train').freeze_transforms()

        rotated_test = AvalancheDataset(
            mnist_test,
            transform_groups=rotation_transforms,
            initial_transform_group='eval').freeze_transforms()

        list_train_dataset.append(rotated_train)
        list_test_dataset.append(rotated_test)

    return nc_benchmark(
        list_train_dataset,
        list_test_dataset,
        n_experiences=len(list_train_dataset),
        task_labels=True,
        shuffle=False,
        class_ids_from_zero_in_each_exp=True,
        one_dataset_per_exp=True,
        train_transform=train_transform,
        eval_transform=eval_transform)


def _get_mnist_dataset(dataset_root):
    if dataset_root is None:
        dataset_root = default_dataset_location('mnist')

    train_set = MNIST(root=dataset_root,
                      train=True, download=True)

    test_set = MNIST(root=dataset_root,
                     train=False, download=True)

    return train_set, test_set


print('Rotated MNIST')
benchmark_instance = RotatedMNIST(
        5, train_transform=None, eval_transform=None)
check_vision_benchmark(benchmark_instance)



