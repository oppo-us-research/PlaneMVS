# Detectors

This directory contains the workflow of training and inference for different models

## PlaneRCNN basic

`planercnn.py` is the planercnn model without mask refinement and the warping loss modules. It can be served as a baseline model to try different ideas.

## PlaneRCNN full and variants

`planercnn_refine.py` is the planercnn full model with mask refinement and warping loss modules. It also contains the consistency loss proposed in [peek-a-boo](https://www.nec-labs.com/~mas/peekaboo/)

## PlaneStereo

`planestereo.py` is the planestereo model for our method. It has three components: planercnn basic detection module, planestereo module and a plane refinement module.
