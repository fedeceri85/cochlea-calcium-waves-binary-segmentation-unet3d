from monai.networks.nets import AttentionUnet, UNet


def build_binary_model(
        architecture,
        spatial_dims,
        in_channels,
        out_channels,
        channels,
        strides,
        num_res_units,
):
    channels = tuple(channels)
    strides = tuple(strides)

    if architecture == "attention_unet":
        if num_res_units is not None:
            raise ValueError("model.num_res_units must be null when model.architecture is attention_unet.")
        return AttentionUnet(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=out_channels,
            channels=channels,
            strides=strides,
        )

    if architecture == "unet":
        if num_res_units is None:
            raise ValueError("model.num_res_units is required when model.architecture is unet.")
        return UNet(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=out_channels,
            channels=channels,
            strides=strides,
            num_res_units=num_res_units,
        )

    raise ValueError(f"Unknown model.architecture: {architecture}")