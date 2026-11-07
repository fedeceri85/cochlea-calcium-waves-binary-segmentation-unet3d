from monai.inferers import sliding_window_inference
from monai.transforms import ScaleIntensity, Activations, AsDiscrete, Compose


import math
import torch
import time

def predict_binary(
    model,
    image,
    device,
    sw_batch_size,
    prediction_sigmoid,
    prediction_threshold,
    roi_size,
):

    # MONAI uses the input dtype for its sliding-window accumulation buffer.
    # Raw microscopy movies are commonly uint16, but the model produces float
    # logits; accumulating those logits into a uint16 buffer raises a cast
    # error. Keep the full movie on the CPU while converting only its dtype.
    image = image.to(device="cpu", dtype=torch.float32)

    post_pred = Compose([
        Activations(sigmoid=prediction_sigmoid),
        AsDiscrete(threshold=prediction_threshold),
    ])

    # Same normalisation used for individual training patches.
    scale_intensity = ScaleIntensity(minv=0.0, maxv=1.0)
    image = scale_intensity(image)
    def predict_windows(windows):
        # windows has shape:
        # # (number_of_windows, channel, time, height, width)
        # #
        # # Scale each window separately, just like during training.
        # windows = torch.stack([
        #     scale_intensity(window)
        #     for window in windows
        # ])

        return model(windows)

    model.eval()

    with torch.no_grad():
        start = time.perf_counter()
        output = sliding_window_inference(
            inputs=image,
            roi_size=tuple(roi_size),
            sw_batch_size=sw_batch_size,
            predictor=predict_windows,

            # Run the model on this device.
            sw_device=device,
            progress = True,
            # Assemble the complete prediction on the CPU.
            device="cpu",
            buffer_steps=20,
            buffer_dim=0,  # time dimension within the spatial dimensions
            overlap=0.5,
            mode="gaussian",
            sigma_scale=0.125,
        )
        inference_seconds = time.perf_counter() - start
        print(f"Inference and assembly: {inference_seconds:.1f}s")

        # The input movie is no longer needed after inference. Release it
        # before allocating temporary tensors for the Dice calculation.

        predictions = post_pred(output)

    return predictions



