import numpy as np
import tensorflow_hub as hub
import mediapy as media
import moviepy.editor as mpy

from tool import (
    Interpolator,
    interpolate_recursively_from_files,
    interpolate_recursively_from_memory,
)

# model = hub.load("https://tfhub.dev/google/film/1")


# # using images from the FILM repository (https://github.com/google-research/frame-interpolation/)

# image_1_url = "https://github.com/google-research/frame-interpolation/blob/main/photos/one.png?raw=true"
# image_2_url = "https://github.com/google-research/frame-interpolation/blob/main/photos/two.png?raw=true"

# time = np.array([0.5], dtype=np.float32)

# image1 = load_image(image_1_url)
# image2 = load_image(image_2_url)

# input = {
#     "time": np.expand_dims(time, axis=0),  # adding the batch dimension to the time
#     "x0": np.expand_dims(image1, axis=0),  # adding the batch dimension to the image
#     "x1": np.expand_dims(image2, axis=0),  # adding the batch dimension to the image
# }
# mid_frame = model(input)

# print(mid_frame.keys())
# frames = [image1, mid_frame["image"][0].numpy(), image2]

# media.write_video("./slow_motion_raw.mp4", frames, fps=3)
# model = hub.load(
#     "https://tfhub.dev/tensorflow/tfgan/estimator/frame_interpolation/3:1/32x32/1"
# )

clip = mpy.VideoFileClip("source.mp4")

print(clip.fps, clip.duration)


# Convertimos el video a un array de numpy
frames = np.array(
    [np.array(clip.get_frame(t)) for t in np.arange(0, clip.duration, 1 / clip.fps)]
)

print('frames.shape', frames.shape)

# # Interpolamos los frames para hacer slow motion 3x
# interpolated = model(frames).numpy()

# # Convertimos el array interpolado nuevamente a clip de video
# slowmo_clip = mpy.ImageSequenceClip(list(interpolated), fps=clip.fps * 3)

# Convertimos el video a un array de numpy
frames = np.array(
    [np.array(clip.get_frame(t)) for t in np.arange(0, clip.duration, 1 / clip.fps)]
)


times_to_interpolate = 6
interpolator = Interpolator()
input_frames = ["./images/one.png", "./images/two.png"]
frames = list(
    interpolate_recursively_from_files(input_frames, times_to_interpolate, interpolator)
)

media.write_video("./slow_motion.mp4", frames, fps=30)
