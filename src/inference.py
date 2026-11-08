from monai.inferers import sliding_window_inference
from monai.transforms import ScaleIntensity, Activations, AsDiscrete, Compose

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

    activation = Activations(sigmoid=prediction_sigmoid)
    threshold = AsDiscrete(threshold=prediction_threshold)



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
        probabilities = activation(output)
        predictions = threshold(probabilities)


    return predictions,probabilities


from scipy.ndimage import binary_closing
import numpy as np
from scipy import ndimage as ndi
from skimage.morphology import remove_small_objects

def reorder_labels_temporally(labels):
    '''
    Reorder labels along the first dimension
    '''
    objects = ndi.find_objects(labels)

    existing_labels = [
        label_id
        for label_id, slices in enumerate(objects, start=1)
        if slices is not None
    ]

    # Sort by first time point, then spatial position for deterministic ties.
    ordered_labels = sorted(
        existing_labels,
        key=lambda label_id: (
            objects[label_id - 1][0].start,  # first time
            objects[label_id - 1][1].start,  # upper y
            objects[label_id - 1][2].start,  # left x
        ),
    )

    lookup = np.zeros(labels.max() + 1, dtype=np.int32)
    lookup[ordered_labels] = np.arange(
        1,
        len(ordered_labels) + 1,
        dtype=np.int32,
    )

    return lookup[labels]

def binary_to_instance_labels(binary_mask, min_size=2000):
    '''Transform a binary mask into single wave instances. Removes ROIs below min_size.
    
    binary_mask: three-dimensional np.array
    '''

    # 1. Ensure boolean mask
    mask = binary_mask > 0

    print("Closing the mask")
    #mask = closing(mask, footprint=np.ones((1,1,4, 4, 4)))
    mask = binary_closing(mask, structure= np.ones((4,4,4)))

    structure = ndi.generate_binary_structure(
        rank=3,
        connectivity=2,
    )

    labels, count = ndi.label(mask, structure=structure)
    labels = remove_small_objects(labels, min_size=min_size)
    print("Reordering labels temporally")
    labels = reorder_labels_temporally(labels)
    return labels
