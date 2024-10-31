import cv2
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def load_video(file_path: str, sub_sample=None, skip_frames=None) -> np.array:
    """
    :param file_path: path to video
    :param sub_sample: Factor to reduce the resolution of each frame.
    :param skip_frames: Number of frames to skip (e.g., skip_frames=1 will process every other frame).
    :return: numpy array (frame_count, frame_height, frame_width, 3) of the frames in the video.
    """
    video_capture = cv2.VideoCapture(file_path)
    frame_count = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Adjust dimensions for subsampling
    if sub_sample:
        frame_width = frame_width // sub_sample
        frame_height = frame_height // sub_sample

    # List to store video frames
    frames = []

    fc = 0
    ret = True
    while ret:
        ret, frame = video_capture.read()
        if not ret:  # Break if no more frames are read
            break

        if skip_frames and fc % (skip_frames + 1) != 0:
            fc += 1
            continue

        if sub_sample:
            blurred_frame = cv2.GaussianBlur(frame, (0, 0), sigmaX=sub_sample / 4, sigmaY=sub_sample / 4)
            frame = blurred_frame[::sub_sample, ::sub_sample][:frame_height, :frame_width]

        frames.append(frame)
        fc += 1

    video_capture.release()

    video = np.array(frames, dtype='uint8')
    print(f"loaded {video.shape[0]} frames")
    return video


def create_video(video, save=False):
    fig = plt.figure()

    frames = []
    for i in range(len(video)):
        frames.append([plt.imshow(video[i], animated=True)])
    ani = animation.ArtistAnimation(fig, frames, blit=True, repeat_delay=100)

    if save:
        ani.save('output.mp4')

    return ani


def dual_video(video1, points, templates, save=False):
    fig = plt.figure(figsize=(8, 8))
    gs = fig.add_gridspec(2, 2, width_ratios=[2, 1], height_ratios=[2, 1])  # Adjust ratios as desired
    plt.subplots_adjust(wspace=0.3, hspace=0.1)

    ax1 = fig.add_subplot(gs[0, :])
    ax2 = fig.add_subplot(gs[1, 0])
    ax3 = fig.add_subplot(gs[1, 1])

    assert len(video1) == len(points)
    frames = []

    for i in range(len(video1)):
        img = ax1.imshow(video1[i])
        x_coords = points[i][0][:, 1]
        y_coords = points[i][0][:, 0]
        sct = ax2.scatter(x_coords, y_coords, c=points[i][1], cmap="Reds")
        # ax2.invert_yaxis()
        templ = ax3.imshow(templates[i].astype(int))
        frames.append([img, sct, templ])

    ani = animation.ArtistAnimation(fig, frames, blit=False, interval=100)

    if save:
        ani.save("animation.gif")

    return ani


def draw_box(image, x, y, window_size_x, window_size_y):
    W, H, _ = image.shape
    window_size_x = window_size_x // 2
    window_size_y = window_size_y // 2

    x_min = int(max(0, x - window_size_x))
    x_max = int(min(W, x + window_size_x))
    y_min = int(max(0, y - window_size_y))
    y_max = int(min(H, y + window_size_y))

    color = [255, 0, 0]  # Red in RGB
    image[x_min:x_max, y_min] = color
    image[x_min:x_max, y_max - 1] = color
    image[x_min, y_min:y_max] = color
    image[x_max - 1, y_min:y_max] = color

    return image


def get_center(template):
    return int(template[0][1] + template[1][1] * 0.5), int(template[0][0] + template[1][0] * 0.5)
