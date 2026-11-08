# 3D U-Net for Cochlear Calcium Wave Segmentation

[![Hugging Face: IHC model](https://img.shields.io/badge/🤗%20Hugging%20Face-IHC%20model-FFD21E)](TODO)
[![Hugging Face: IHC afferent model](https://img.shields.io/badge/🤗%20Hugging%20Face-IHC%20afferent%20model-FFD21E)](TODO)

MONAI/PyTorch pipeline for segmenting calcium waves in the neonatal cochlea. The model is trained on GCaMP-labelled cochleae from in vivo two-photon imaging experiments ([paper](https://www.nature.com/articles/s41467-024-55519-w)).

The full pipeline consists of two steps: initial binary segmentation using a U-Net, followed by post-processing to isolate individual instances and refine the predictions.

## Train

Training requires a config file. Epochs, workers, checkpoint paths, model settings, augmentation flags/probabilities, split settings, and data paths are read from YAML.

```bash
python scripts/train_binary.py config/binary_model_v2_ca_waves.yml
```

Training resumes from the checkpoint epoch when a checkpoint exists; otherwise, it starts from scratch. Training stops at `training.epochs`.

## Inference

The `inference_example.ipynb` notebook provides an example of running calcium-wave inference on a movie.

