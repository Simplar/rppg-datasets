""" Loaders of data from the rPPG (Remote Photoplethysmography) datasets """

import importlib

from .ds_title import DSTitle
from .loader_base import DatasetLoader


def import_loader(class_name_postfix: str) -> DatasetLoader.__class__:
    """
    Convinience import of dataset loader by its short name.
    :param class_name_postfix: postfix of name of the dataset loader, e.g. `dccsfedu`.
    :return: Class of selected dataset loader.
    """
    module_name_rel = f"loader_{class_name_postfix[: -len('DatasetLoader')].lower()}"
    module_name_abs = f"{__name__}.{module_name_rel}"
    loader_module = importlib.import_module(module_name_abs, package=None)
    cls = loader_module.__dict__[class_name_postfix]
    return cls
