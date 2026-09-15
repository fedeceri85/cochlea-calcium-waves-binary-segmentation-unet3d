# Chunked reading/writing with optional MONAI Gaussian smoothing and OOF frame replacement
import os
import shutil
from os.path import getsize
from pathlib import Path

import dask.array as da
import numpy as np
import pandas as pd
import tifffile as tf
import torch
import zarr
from monai.transforms import Compose, EnsureTyped, GaussianSmoothd
from ome_zarr.format import FormatV05
from ome_zarr.writer import write_image
from tqdm.auto import tqdm

PREVIEWFILENAME = 'ChanC_Preview.tif'
PREVIEWFILENAME2 = 'ChanA_Preview.tif'
MAXCHUNKSIZE = 1024 * 288 * 2000 # Change this depenging on the available VRAM.
FILENAME = 'Image_001_001.raw'


def getImgInfo(folder, filename=FILENAME, n_channels=1):
    fullpath = os.path.join(folder, filename)

    try:
        prev = tf.imread(os.path.join(folder, PREVIEWFILENAME))
    except FileNotFoundError:
        prev = tf.imread(os.path.join(folder, PREVIEWFILENAME2))

    height, width = prev.shape
    frame_size = width * height * np.dtype(np.uint16).itemsize
    n_frames = int(getsize(fullpath) / (frame_size * n_channels))
    return width, height, n_frames


def process_stack(
    reader,
    height,
    width,
    n_frames,
    n_channels=1,
    start=0,
    end=-1,
    pbar=None,
    show_progress=True,
    progress_desc=None,
    channel=1,
    output_path=None,
    return_stack=None,
    output_dtype=None,
    chunk_size_frames=None,
    apply_gaussian=False,
    spatial_gaussian_sigma=0.0,
    temporal_gaussian_sigma=0.0,
    frame_size=None,
    processing_params=None,
    oof_dataframe=None,
    oof_frame_column='Frame',
    oof_label_column='Label',
    oof_bad_label=1,
    zarr_chunks=None,
    zarr_shards=None,
    ome_zarr_scale=None,
    overwrite=True,
    device='auto',
):
    """Read one raw movie channel in chunks, optionally smooth, blank OOF frames, and write output.

    If processing_params is passed, first_frame/last_frame are read from the JSON.
    first_frame is treated as 1-based and converted to a 0-based raw offset here.

    output_path=None means do not write anything.
    output_path ending in .tif/.tiff writes BigTIFF.
    output_path ending in .zarr/.ome.zarr writes OME-Zarr with ome_zarr.writer.write_image.
    """
    height = int(height)
    width = int(width)
    n_frames = int(n_frames)

    if processing_params is not None:
        n_channels = int(processing_params['nChannels'])
        spatial_gaussian_sigma = float(processing_params['spatialGaussian'])
        temporal_gaussian_sigma = float(processing_params['temporalGaussian'])
        raw_offset = int(processing_params['first_frame']) - 1
        available_frames = int(processing_params['last_frame']) - int(processing_params['first_frame']) + 1
    else:
        n_channels = int(n_channels)
        raw_offset = 0
        available_frames = n_frames

    if end == -1:
        end = available_frames

    source_start = raw_offset + int(start)
    source_end = raw_offset + int(end)
    total_frames = int(end) - int(start)

    if frame_size is None:
        frame_size = height * width * np.dtype(np.uint16).itemsize

    if chunk_size_frames is None:
        chunk_size_frames = max(1, MAXCHUNKSIZE // (frame_size * n_channels))
    chunk_size_frames = int(chunk_size_frames)

    if return_stack is None:
        return_stack = output_path is None

    if output_dtype is None:
        output_dtype = np.float32 if apply_gaussian else np.uint16
    output_dtype = np.dtype(output_dtype)

    if device == 'auto':
        if torch.cuda.is_available():
            torch_device = torch.device('cuda')
            print(f'Using CUDA device: {torch.cuda.get_device_name(torch.cuda.current_device())}')
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            torch_device = torch.device('mps')
            print('Using MPS device.')
        else:
            torch_device = torch.device('cpu')
            print('Using CPU device.')
    else:
        torch_device = torch.device(device)
        print(f'Using user-specified device: {torch_device}')

    gaussian = None
    temporal_halo = 0
    if apply_gaussian and (spatial_gaussian_sigma > 0 or temporal_gaussian_sigma > 0):
        gaussian = Compose([
            EnsureTyped(keys=['image'], dtype=torch.float32, device=torch_device, track_meta=False),
            GaussianSmoothd(keys=['image'], sigma=(temporal_gaussian_sigma, spatial_gaussian_sigma, spatial_gaussian_sigma)),
        ])
        temporal_halo = int(4.0 * temporal_gaussian_sigma + 0.5)

    oof_labels = None
    if oof_dataframe is not None:
        # The CSV Frame values use the original 1-based movie numbering, so subtract 1 once.
        oof_labels = dict(zip(
            oof_dataframe[oof_frame_column].astype(int).to_numpy() - 1,
            oof_dataframe[oof_label_column].astype(int).to_numpy(),
        ))

    output = None
    output_kind = None
    temp_path = None
    output_path = None if output_path is None else Path(output_path)
    if output_path is not None:
        suffixes = ''.join(output_path.suffixes).lower()
        if suffixes.endswith(('.tif', '.tiff')):
            if output_path.exists():
                if not overwrite:
                    return output_path
                output_path.unlink()
            output = tf.memmap(output_path, shape=(total_frames, height, width), dtype=output_dtype, bigtiff=True)
            output_kind = 'tiff'
        elif suffixes.endswith(('.zarr', '.ome.zarr')):
            temp_path = output_path.with_name(output_path.name + '.tmp')
            if temp_path.exists():
                shutil.rmtree(temp_path) if temp_path.is_dir() else temp_path.unlink()
            if output_path.exists():
                if not overwrite:
                    return output_path
                shutil.rmtree(output_path) if output_path.is_dir() else output_path.unlink()

            if zarr_chunks is None:
                zarr_chunks = (min(chunk_size_frames, total_frames), min(128, height), min(128, width))
            storage_options = {'chunks': zarr_chunks}
            if zarr_shards is not None:
                storage_options['shards'] = zarr_shards

            try:
                empty = da.empty((total_frames, height, width), chunks=zarr_chunks, dtype=output_dtype)
                write_image(
                    empty,
                    str(temp_path),
                    axes='tyx',
                    scale=ome_zarr_scale,
                    scale_factors=[],
                    fmt=FormatV05(),
                    storage_options=storage_options,
                    compute=False,
                )
                output = zarr.open(str(temp_path), mode='r+')['s0']
            except Exception:
                if temp_path.exists():
                    shutil.rmtree(temp_path) if temp_path.is_dir() else temp_path.unlink()
                raise
            output_kind = 'ome-zarr'
        else:
            raise ValueError('output_path must end in .tif, .tiff, .zarr, or .ome.zarr')

    chunks = []
    written = 0
    last_good_frame = None
    progress = tqdm(total=total_frames, desc=progress_desc or 'loadWholeStack', unit='frame') if show_progress else None

    success = False
    try:
        for write_start in range(source_start, source_end, chunk_size_frames):
            write_end = min(write_start + chunk_size_frames, source_end)

            read_start = max(raw_offset, write_start - temporal_halo)
            read_end = min(raw_offset + available_frames, write_end + temporal_halo)
            read_count = read_end - read_start

            # Raw frames are assumed to be interleaved by channel:
            # t0_c1, t0_c2, t1_c1, t1_c2, ...
            reader.seek((read_start * n_channels + (channel - 1)) * frame_size)
            raw_frames_to_read = (read_count - 1) * n_channels + 1
            raw = np.frombuffer(reader.read(frame_size * raw_frames_to_read), dtype=np.uint16)
            raw = raw.reshape((-1, height, width))
            stack = raw[::n_channels]

            if gaussian is not None:
                with torch.no_grad():
                    stack = gaussian({'image': stack[np.newaxis, ...]})['image'][0].detach().cpu().numpy()

            trim_start = write_start - read_start
            trim_end = trim_start + (write_end - write_start)
            stack = stack[trim_start:trim_end]

            if oof_labels is not None:
                stack = np.array(stack, copy=True)
                for i, raw_frame in enumerate(range(write_start, write_end)):
                    if oof_labels.get(raw_frame, 0) == oof_bad_label:
                        if last_good_frame is not None:
                            stack[i] = last_good_frame
                    else:
                        last_good_frame = np.array(stack[i], copy=True)

            if output is not None:
                out_stack = stack
                if np.issubdtype(output_dtype, np.integer):
                    info = np.iinfo(output_dtype)
                    out_stack = np.rint(np.clip(out_stack, info.min, info.max))
                output[written:written + stack.shape[0]] = out_stack.astype(output_dtype, copy=False)
                if output_kind == 'tiff':
                    output.flush()

            if return_stack:
                chunks.append(np.asarray(stack))

            frames_done = write_end - write_start
            written += frames_done
            if pbar is not None:
                pbar.value += frames_done
            if progress is not None:
                progress.update(frames_done)

        success = True

    except Exception:
        if output_kind == 'ome-zarr' and temp_path is not None and temp_path.exists():
            shutil.rmtree(temp_path) if temp_path.is_dir() else temp_path.unlink()
        raise

    finally:
        reader.seek(0)
        if output is not None and output_kind == 'tiff':
            output.flush()
        if progress is not None:
            progress.close()
        if torch_device.type == 'cuda':
            torch.cuda.empty_cache()
        elif torch_device.type == 'mps' and hasattr(torch, 'mps'):
            torch.mps.empty_cache()

    if success and output_kind == 'ome-zarr' and temp_path is not None:
        temp_path.rename(output_path)

    if output_path is not None and not return_stack:
        return output_path
    return chunks
