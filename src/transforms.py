from monai.transforms import MapTransform, EnsureChannelFirstd, ScaleIntensityd, EnsureTyped, RandomizableTransform, Compose,RandFlipd, RandRotate90d, RandScaleIntensityd, RandShiftIntensityd, DeleteItemsd
import zarr
import tifffile
from numpy.random import randint 
import numpy as np
import torch

class InitMoviePatchd(MapTransform):
    def __init__(self, materialize=False):
        super().__init__(keys=["image", "label"])
        self.materialize = materialize # Whether to load full images in disk

    def __call__(self, data)->dict:
        d= dict(data)


        image = zarr.open(d["image"], mode="r")["s0"]
        label = tifffile.memmap(d["label"], mode='r')
        
        if image.shape != label.shape:
            raise ValueError(
                f"Image and label shapes differ: "
                f"{image.shape} vs {label.shape}"
            )

        # Load waves positions
        waves_dict = {}
        with open(d['waves']) as f:
            lines = f.readlines()
            for l in lines:
                l = l.strip('\n')
                wave_id = int(l.split(' (')[0])
                bbox2 = (l.split(' (')[1]).strip(')')
                bbox = [int(i) for i in bbox2.split(',')]
                waves_dict[wave_id] = bbox
        if self.materialize:
            d["image"] = image[:]
            d["label"] = np.array(label, copy=True)
            d["waves"] = waves_dict
        else:
            d["image"] = image
            d["label"] = label
            d["waves"] = waves_dict
        return d

class RandLoadWavePatchd(RandomizableTransform, MapTransform):
    """Load one balanced patch, keeping image and instance labels aligned.

    Foreground sampling chooses a wave uniformly, then a point in its bbox.
    Points outside that wave are rejected using the memory-mapped label.
    Background patches have at most background_max_fraction foreground.
    Failed sampling raises rather than silently changing the requested balance.
    Marked randomizable so MONAI CacheDataset does not cache a sampled patch.
    """

    def __init__(
        self,
        patch_size=(32, 128, 128),
        foreground_probability=0.75,
        min_waves=1,
        min_foreground_fraction=0.0,
        background_max_fraction=0.0,
        max_attempts=100,
        image_key="image",
        label_key="label",
        index_key="waves",
    ):
        RandomizableTransform.__init__(self, prob=1.0)
        MapTransform.__init__(self, keys=[image_key, label_key, index_key])
        self.patch_size = tuple(int(v) for v in patch_size)
        if len(self.patch_size) != 3 or min(self.patch_size) <= 0:
            raise ValueError("patch_size must contain three positive integers.")
        for value in (foreground_probability, min_foreground_fraction,
                      background_max_fraction):
            if not 0 <= value <= 1:
                raise ValueError("Probabilities and fractions must be in [0, 1].")
        if min_waves < 1 or max_attempts < 1:
            raise ValueError("min_waves and max_attempts must be positive.")
        self.foreground_probability = foreground_probability
        self.min_waves = int(min_waves)
        self.min_foreground_fraction = min_foreground_fraction
        self.background_max_fraction = background_max_fraction
        self.max_attempts = int(max_attempts)
        self.image_key, self.label_key, self.index_key = image_key, label_key, index_key

    def randomize(self, data=None):
        self._foreground = self.R.random_sample() < self.foreground_probability

    def _find_foreground_coordinate(self, label, wave_id, box):
        """Choose an actual wave voxel without materialising its full bbox.

        Scan one time plane at a time and use weighted reservoir sampling so
        every foreground voxel in the bounding box has equal probability of
        being selected.
        """
        selected = None
        foreground_seen = 0
        (t0, t1), (y0, y1), (x0, x1) = box

        for t in range(int(t0), int(t1)):
            plane = np.asarray(label[t, y0:y1, x0:x1])
            coordinates = np.argwhere(plane == int(wave_id))
            count = len(coordinates)
            if count == 0:
                continue

            new_total = foreground_seen + count
            if self.R.random_sample() < count / new_total:
                yx = coordinates[self.R.randint(count)]
                selected = (int(t), int(y0 + yx[0]), int(x0 + yx[1]))
            foreground_seen = new_total

        return selected

    def _return_patch(self, d, image, label, starts):
        selection = tuple(
            slice(start, start + size)
            for start, size in zip(starts, self.patch_size)
        )
        label_patch = np.array(label[selection], copy=True)
        fraction = np.count_nonzero(label_patch > 0) / label_patch.size
        ids = np.unique(label_patch)
        ids = ids[ids > 0]

        if self._foreground:
            valid = len(ids) >= self.min_waves and fraction >= self.min_foreground_fraction
        else:
            valid = fraction <= self.background_max_fraction
        if not valid:
            return None

        d[self.image_key] = image[selection]
        d[self.label_key] = label_patch
        d["patch_start"] = starts
        d["patch_is_foreground"] = self._foreground
        d.pop(self.index_key, None)
        return d

    def __call__(self, data):
        d = dict(data)
        image = d[self.image_key]
        label = d[self.label_key]
        index = d[self.index_key]
        shape = tuple(image.shape)
        if len(shape) != 3 or shape != label.shape:
            raise ValueError("Image and label must have matching (time, y, x) shapes.")
        if any(size > dim for size, dim in zip(self.patch_size, shape)):
            raise ValueError("Movie is smaller than patch_size; pad it or use smaller patches.")
        wave_ids = list(index)
        boxes = []
        for wave_id in wave_ids:
            bbox = np.asarray(index[wave_id])
            if bbox.shape != (6,) or not np.issubdtype(bbox.dtype, np.integer):
                raise ValueError("Each bounding box must contain six integer coordinates.")
            box = np.stack((bbox[:3], bbox[3:]), axis=1)
            if (not np.issubdtype(box.dtype, np.integer)
                    or np.any(box[:, 0] < 0) or np.any(box[:, 1] > shape)
                    or np.any(box[:, 0] >= box[:, 1]) or int(wave_id) <= 0):
                raise ValueError("Invalid wave ID or exclusive (time, y, x) bounding box.")
            boxes.append(box)
        self.randomize()
        if not wave_ids:
            self._foreground = False
        if self._foreground and len(wave_ids) < self.min_waves:
            raise ValueError("Index contains fewer waves than min_waves.")

        # Fix the target wave across retries so small/sparse waves aren't
        # systematically discarded in favour of easier waves.
        wave_number = self.R.randint(len(wave_ids)) if self._foreground else None
        for _ in range(self.max_attempts):
            if self._foreground:
                wave_id = wave_ids[wave_number]
                center = tuple(self.R.randint(lo, hi) for lo, hi in boxes[wave_number])
                if label[center] != int(wave_id):
                    continue
                starts = tuple(
                    int(np.clip(c - size // 2, 0, dim - size))
                    for c, size, dim in zip(center, self.patch_size, shape)
                )
            else:
                starts = tuple(self.R.randint(dim - size + 1)
                               for dim, size in zip(shape, self.patch_size))
            result = self._return_patch(d, image, label, starts)
            if result is not None:
                return result

        # Sparse waves can make random rejection sampling unreliable. Fall
        # back to an exact, plane-wise search for a voxel of the selected wave.
        if self._foreground:
            wave_id = wave_ids[wave_number]
            center = self._find_foreground_coordinate(
                label, wave_id, boxes[wave_number]
            )
            if center is None:
                raise RuntimeError(
                    f"Wave {wave_id} has no matching label voxels inside "
                    f"its bounding box {boxes[wave_number].tolist()}. "
                    "Regenerate 3Droi_bboxes.txt from the current label TIFF."
                )

            starts = tuple(
                int(np.clip(c - size // 2, 0, dim - size))
                for c, size, dim in zip(center, self.patch_size, shape)
            )
            result = self._return_patch(d, image, label, starts)
            if result is not None:
                return result

        kind = "foreground" if self._foreground else "background"
        raise RuntimeError(
            f"No valid {kind} patch after {self.max_attempts} attempts. "
            "Check the index, relax occupancy/min_waves constraints, or save "
            "actual foreground coordinates for sparse bounding boxes."
        )

class BinarizeMaskd(MapTransform):

    def __init__(self, keys, dtype=np.float32):
        super().__init__(keys)
        self.dtype = dtype

    def __call__(self, data):
        d = dict(data)

        for key in self.keys:
            d[key] = (d[key] > 0).astype(self.dtype, copy=False)

        return d


def get_transforms_binary_classification(
    patch_size, 
    foreground_probability,
    min_waves,
    min_foreground_fraction,
    background_max_fraction,
    max_attempts,
    flip_vertical,
    flip_vertical_prob,
    flip_horizontal,
    flip_horizontal_prob,
    rotate90,
    rotate90_prob,
    rotate90_max_k,
    intensity_scale,
    intensity_scale_factors,
    intensity_scale_prob,
    intensity_shift,
    intensity_shift_offsets,
    intensity_shift_prob
):
    
    transforms_list = [
            InitMoviePatchd(),
            RandLoadWavePatchd(patch_size=patch_size, foreground_probability=foreground_probability, min_waves=min_waves, min_foreground_fraction=min_foreground_fraction,
                                background_max_fraction=background_max_fraction, max_attempts=max_attempts),
            BinarizeMaskd(keys = ["label"]),
            EnsureChannelFirstd(keys=["image", "label"],channel_dim = "no_channel"),
            ScaleIntensityd(keys="image"),
            EnsureTyped(keys=["image", "label"], dtype=torch.float32),
        ]

    #Data augmentation
    if flip_vertical:
        transforms_list.append( 
        RandFlipd(
            keys=["image", "label"],
            prob=flip_vertical_prob,
            spatial_axis=1,
        ))

    if flip_horizontal:
        transforms_list.append(
        RandFlipd(
            keys=["image", "label"],
            prob=flip_horizontal_prob,
            spatial_axis=2,
        ))
    # Odd quarter-turns swap height and width, so require square patches.
    if rotate90 and patch_size[1] == patch_size[2]:
        transforms_list.append( 
            RandRotate90d(
                keys=["image", "label"],
                prob=rotate90_prob,
                max_k=rotate90_max_k,
                spatial_axes = (1,2)
            ))
    if intensity_scale:
        transforms_list.append(
            RandScaleIntensityd(
                keys=["image"],
                factors=intensity_scale_factors,
                prob=intensity_scale_prob,
            ))
    if intensity_shift:
        transforms_list.append(
            RandShiftIntensityd(
                keys=["image"],
                offsets=intensity_shift_offsets,
                prob=intensity_shift_prob,
            ))

    train_transforms = Compose(transforms_list)

    val_transforms_list = [

        InitMoviePatchd(materialize = True),
        DeleteItemsd(keys=["waves"]),
        BinarizeMaskd(keys=["label"], dtype=np.bool_),
        EnsureChannelFirstd(keys=["image", "label"],channel_dim = "no_channel"),
        #ScaleIntensityd(keys="image"), add scaleintensity to sliding_window_inference
        EnsureTyped(keys=["image"], dtype=torch.float32),
        EnsureTyped(keys=["label"], dtype=torch.bool),
    ]

    val_transforms = Compose(val_transforms_list)    

    return train_transforms, val_transforms
