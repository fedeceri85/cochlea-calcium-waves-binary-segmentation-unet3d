from monai.inferers import sliding_window_inference
from monai.transforms import Activations, AsDiscrete, Compose, ScaleIntensity
import torch


def validate_binary(
    model,
    loader,
    dice_metric,
    device,
    prediction_sigmoid,
    prediction_threshold,
    label_threshold,
    roi_size,
    sw_batch_size,
):
    post_pred = Compose([
        Activations(sigmoid=prediction_sigmoid),
        AsDiscrete(threshold=prediction_threshold),
    ])
    post_label = AsDiscrete(threshold=label_threshold)

    # Same normalisation used for individual training patches.
    scale_intensity = ScaleIntensity(minv=0.0, maxv=1.0)

    def predict_windows(windows):
        # windows has shape:
        # (number_of_windows, channel, time, height, width)
        #
        # Scale each window separately, just like during training.
        windows = torch.stack([
            scale_intensity(window)
            for window in windows
        ])

        return model(windows)

    model.eval()
    dice_metric.reset()

    with torch.no_grad():
        for batch in loader:
            # Keep the complete movie and its label on the CPU.
            image = batch["image"].to("cpu")
            label = batch["label"].to("cpu")

            output = sliding_window_inference(
                inputs=image,
                roi_size=tuple(roi_size),
                sw_batch_size=sw_batch_size,
                predictor=predict_windows,

                # Run the model on this device.
                sw_device=device,

                # Assemble the complete prediction on the CPU.
                device="cpu",
            )

            pred = post_pred(output)
            gt = post_label(label)

            dice_metric(pred, gt)

    score = dice_metric.aggregate().item()
    dice_metric.reset()

    return score