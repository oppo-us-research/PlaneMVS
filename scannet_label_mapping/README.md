# Scannet Label Mapping

This directory contains semantic label mapping relationships from scannet, to nyu, then to 11 classes we define for planar stuff or objects.

## Mapping

We first find the original scannet class for each plane from `scannetv2-labels.combined.tsv`, then we build mapping from scannet classes to nyu classes and save into a json file `scannet_nyu_map.json`. Then we combine and filter some classes to form the final 11 classes we use for planar region, during training and testing, and save it into `canonical_mapping.json`.

## Labels

You can take a look at the final mapping between canonical id to class label `labels.txt`
