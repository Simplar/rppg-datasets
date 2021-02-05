""" Loaders of data from the rPPG (Remote Photoplethysmography) datasets """

import importlib
import os

from .loader_base import DatasetLoader
from .ds_title import DSTitle


def import_loader(class_name_postfix: str) -> DatasetLoader.__class__:
    """
    Convinience import of dataset loader by its short name.
    :param class_name_postfix: postfix of name of the dataset loader, e.g. `dccsfedu`.
    :return: Class of selected dataset loader.
    """
    rel_path = os.path.dirname(os.path.relpath(__file__, 'loader'))
    rel_path_dots = rel_path.replace('\\', '.')
    module_name = f"{rel_path_dots}.loader_{class_name_postfix[len(''): -len('DatasetLoader')].lower()}"
    loader_module = importlib.import_module(module_name, package=None)
    cls = loader_module.__dict__[class_name_postfix]
    return cls
