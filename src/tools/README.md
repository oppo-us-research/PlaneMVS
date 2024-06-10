# Tools

This directory contains the entrances of training and inference(evaluation) scripts.

## Training models on scannet

`train_net.py` is the main function for all training scripts. By default, it trains the model by assigning epoch numbers and do validation every 1000 iterations.

## Inference on scannet

`inference.py` is the main function of inference and evaluation for planercnn(and its variants) on scannet dataset. It both evaluates plane detection and plane geometry.

`stereo_inference.py` is the main function of inference and evaluation for planestereo on scannet dataset.

## Inference on 7-scenes

`seven_scenes_inference.py` is the main function of inference and evaluation for planestereo and planercnn on unseen 7-scenes dataset. Since 7-scenes does not contain plane groundtruths, we directly test the models on it to compare the generalizability.
