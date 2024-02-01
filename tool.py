"""A wrapper class for running a frame interpolation based on the FILM model on TFHub

Usage:
  interpolator = Interpolator()
  result_batch = interpolator(image_batch_0, image_batch_1, batch_dt)
  Where image_batch_1 and image_batch_2 are numpy tensors with TF standard
  (B,H,W,C) layout, batch_dt is the sub-frame time in range [0..1], (B,) layout.
"""


from typing import Generator, List, Optional, Iterable
import numpy as np
import requests
import tensorflow as tf
import tensorflow_hub as hub
from tqdm import tqdm

_UINT8_MAX_F = float(np.iinfo(np.uint8).max)
_CONFIG_FFMPEG_NAME_OR_PATH = "ffmpeg"


def load_image(img_url: str):
    """Returns an image with shape [height, width, num_channels], with pixels in [0..1] range, and type np.float32."""

    if img_url.startswith("https"):
        user_agent = {"User-agent": "Colab Sample (https://tensorflow.org)"}
        response = requests.get(img_url, headers=user_agent)
        image_data = response.content
    else:
        image_data = tf.io.read_file(img_url)

    image = tf.io.decode_image(image_data, channels=3)
    image_numpy = tf.cast(image, dtype=tf.float32).numpy()
    return image_numpy / _UINT8_MAX_F


def read_image(filename: str) -> np.ndarray:
    """Reads an sRgb 8-bit image.

    Args:
      filename: The input filename to read.

    Returns:
      A float32 3-channel (RGB) ndarray with colors in the [0..1] range.
    """
    image_data = tf.io.read_file(filename)
    image = tf.io.decode_image(image_data, channels=3)
    image_numpy = tf.cast(image, dtype=tf.float32).numpy()
    return image_numpy / _UINT8_MAX_F


def _pad_to_align(x, align):
    """Pads image batch x so width and height divide by align.

    Args:
    x: Image batch to align.
    align: Number to align to.

    Returns:
    1) An image padded so width % align == 0 and height % align == 0.
    2) A bounding box that can be fed readily to tf.image.crop_to_bounding_box
        to undo the padding.
    """
    # Input checking.
    assert np.ndim(x) == 4
    assert align > 0, "align must be a positive number."

    height, width = x.shape[-3:-1]
    height_to_pad = (align - height % align) if height % align != 0 else 0
    width_to_pad = (align - width % align) if width % align != 0 else 0

    bbox_to_pad = {
        "offset_height": height_to_pad // 2,
        "offset_width": width_to_pad // 2,
        "target_height": height + height_to_pad,
        "target_width": width + width_to_pad,
    }
    padded_x = tf.image.pad_to_bounding_box(x, **bbox_to_pad)
    bbox_to_crop = {
        "offset_height": height_to_pad // 2,
        "offset_width": width_to_pad // 2,
        "target_height": height,
        "target_width": width,
    }
    return padded_x, bbox_to_crop


class Interpolator:
    """A class for generating interpolated frames between two input frames.

    Uses the Film model from TFHub
    """

    def __init__(self, align: int = 64) -> None:
        """Loads a saved model.

        Args:
            align: 'If >1, pad the input size so it divides with this before
            inference.'
        """
        self._model = hub.load("https://tfhub.dev/google/film/1")
        self._align = align

    def __call__(self, x0: np.ndarray, x1: np.ndarray, dt: np.ndarray) -> np.ndarray:
        """Generates an interpolated frame between given two batches of frames.

        All inputs should be np.float32 datatype.

        Args:
            x0: First image batch. Dimensions: (batch_size, height, width, channels)
            x1: Second image batch. Dimensions: (batch_size, height, width, channels)
            dt: Sub-frame time. Range [0,1]. Dimensions: (batch_size,)

        Returns:
            The result with dimensions (batch_size, height, width, channels).
        """
        if self._align is not None:
            x0, bbox_to_crop = _pad_to_align(x0, self._align)
            x1, _ = _pad_to_align(x1, self._align)

        inputs = {"x0": x0, "x1": x1, "time": dt[..., np.newaxis]}
        result = self._model(inputs, training=False)
        image = result["image"]

        if self._align is not None:
            image = tf.image.crop_to_bounding_box(image, **bbox_to_crop)
        return image.numpy()


def _recursive_generator(
    frame1: np.ndarray,
    frame2: np.ndarray,
    num_recursions: int,
    interpolator: Interpolator,
    bar: Optional[tqdm] = None,
) -> Generator[np.ndarray, None, None]:
    """Splits halfway to repeatedly generate more frames.

    Args:
      frame1: Input image 1.
      frame2: Input image 2.
      num_recursions: How many times to interpolate the consecutive image pairs.
      interpolator: The frame interpolator instance.

    Yields:
      The interpolated frames, including the first frame (frame1), but excluding
      the final frame2.
    """
    if num_recursions == 0:
        yield frame1
    else:
        # Adds the batch dimension to all inputs before calling the interpolator,
        # and remove it afterwards.
        time = np.full(shape=(1,), fill_value=0.5, dtype=np.float32)
        mid_frame = interpolator(
            frame1[np.newaxis, ...], frame2[np.newaxis, ...], time
        )[0]
        bar.update(1) if bar is not None else bar
        yield from _recursive_generator(
            frame1, mid_frame, num_recursions - 1, interpolator, bar
        )
        yield from _recursive_generator(
            mid_frame, frame2, num_recursions - 1, interpolator, bar
        )


def interpolate_recursively_from_files(
    frames: List[str],
    times_to_interpolate: int,
    interpolator: Interpolator,
) -> Iterable[np.ndarray]:
    """Generates interpolated frames by repeatedly interpolating the midpoint.

    Loads the files on demand and uses the yield paradigm to return the frames
    to allow streamed processing of longer videos.

    Recursive interpolation is useful if the interpolator is trained to predict
    frames at midpoint only and is thus expected to perform poorly elsewhere.

    Args:
      frames: List of input frames. Expected shape (H, W, 3). The colors should be
        in the range[0, 1] and in gamma space.
      times_to_interpolate: Number of times to do recursive midpoint
        interpolation.
      interpolator: The frame interpolation model to use.

    Yields:
      The interpolated frames (including the inputs).
    """
    n = len(frames)
    num_frames = (n - 1) * (2 ** (times_to_interpolate) - 1)
    bar = tqdm(total=num_frames, ncols=100, colour="green")
    for i in range(1, n):
        yield from _recursive_generator(
            read_image(frames[i - 1]),
            read_image(frames[i]),
            times_to_interpolate,
            interpolator,
            bar,
        )
    # Separately yield the final frame.
    yield read_image(frames[-1])


def interpolate_recursively_from_memory(
    frames: List[np.ndarray],
    times_to_interpolate: int,
    interpolator: Interpolator,
) -> Iterable[np.ndarray]:
    """Generates interpolated frames by repeatedly interpolating the midpoint.

    This is functionally equivalent to interpolate_recursively_from_files(), but
    expects the inputs frames in memory, instead of loading them on demand.

    Recursive interpolation is useful if the interpolator is trained to predict
    frames at midpoint only and is thus expected to perform poorly elsewhere.

    Args:
      frames: List of input frames. Expected shape (H, W, 3). The colors should be
        in the range[0, 1] and in gamma space.
      times_to_interpolate: Number of times to do recursive midpoint
        interpolation.
      interpolator: The frame interpolation model to use.

    Yields:
      The interpolated frames (including the inputs).
    """
    n = len(frames)
    num_frames = (n - 1) * (2 ** (times_to_interpolate) - 1)
    bar = tqdm(total=num_frames, ncols=100, colour="green")
    for i in range(1, n):
        yield from _recursive_generator(
            frames[i - 1], frames[i], times_to_interpolate, interpolator, bar
        )
    # Separately yield the final frame.
    yield frames[-1]
