# Copyright (c) Facebook, Inc. and its affiliates.
from d2.utils.registry import Registry

DATASET_REGISTRY = Registry("DATASET")
DATASET_REGISTRY.__doc__ = """
Registry for backbones, which extract feature maps from images

The registered object must be a callable that accepts two arguments:

1. A :class:`detectron2.config.CfgNode`
2. A :class:`detectron2.layers.ShapeSpec`, which contains the input shape specification.

Registered object must return instance of :class:`Backbone`.
"""


def build_dataset(cfg, split):
    """
    Build a backbone from `cfg.DATASET.NAME`.

    Returns:
        an instance of :class:`Dataset`
    """
    dataset_name = cfg.DATASET.NAME
    dataset = DATASET_REGISTRY.get(dataset_name)(cfg, split)
    return dataset