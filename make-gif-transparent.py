#!/usr/bin/env python
import os

DEBUG = False


def run(path_gif_in, path_gif_out=None, threshold=5):
    import cv2
    import numpy as np
    import scipy.stats

    if path_gif_out is None:
        path_gif_base, path_gif_ext = os.path.splitext(path_gif_in)
        path_gif_out = (path_gif_base + '-transparent' +
                        (path_gif_ext or '.gif'))

    frames = []
    # open GIF
    cap = cv2.VideoCapture(path_gif_in)
    fps = cap.get(cv2.CAP_PROP_FPS)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # store frame
        frames.append(frame)
    # release file
    cap.release()

    frames = np.asarray(frames)  # N x H x W x C
    border_pixels = border_size_heuristic(frames)
    border = extract_border(frames, border_pixels)

    # get most common pixel value (infer as background color)
    background_color = scipy.stats.mode(border).mode.squeeze(axis=0)  # noqa
    N, H, W, C = frames.shape  # noqa
    new_value = ([0, 0, 255, 255] if DEBUG else [0, 0, 0, 0])
    flags = 4 | (255 << 8) | cv2.FLOODFILL_MASK_ONLY
    # diff = [20] * 3
    diff = [threshold] * 3
    # flood filling time for each frame
    frames_out = []
    for frame in frames:
        where_h, where_w = where_border_equals_color(
            frame, background_color, border_pixels)
        if C == 4:
            # opencv cannot operate on views - must have a copy
            frame_no_alpha = frame[:, :, :-1].copy()
            frame_alpha = frame
        else:
            frame_no_alpha = frame
            # add alpha channel
            frame_alpha = cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA)
        mask = np.zeros((H + 2, W + 2), dtype=np.uint8)
        for h, w in zip(where_h, where_w):
            if mask[h, w]:
                continue
            _, _, mask, _ = cv2.floodFill(
                frame_no_alpha,
                mask=mask,
                seedPoint=(w, h),
                newVal=new_value[:2],
                loDiff=diff,
                upDiff=diff,
                flags=flags,
            )
        frame_alpha[mask[1:-1, 1:-1] == 255] = new_value
        frames_out.append(frame_alpha)
    # write out GIF
    save_gif(frames_out, fps, path_gif_out)


def save_gif(frames, fps, path_gif_out):
    import subprocess
    import tempfile
    import math
    import os
    import cv2
    import shutil

    # convert to centiseconds
    delay = 100 / fps

    # save frames to temporary directory
    dirname = tempfile.mkdtemp()
    n_digits = math.ceil(math.log10(len(frames) + 1))
    for i, frame in enumerate(frames):
        cv2.imwrite(os.path.join(dirname, f'FRAME_{i:0{n_digits}d}.png'),
                    frame)

    try:
        # ImageMagick time
        subprocess.run(
            ['convert',
             '-delay', str(delay),
             '-loop', '0',
             '-alpha', 'set',
             '-dispose', 'previous',
             os.path.join(dirname, 'FRAME_*.png'),
             path_gif_out]
        )
    finally:
        shutil.rmtree(dirname)


def border_size_heuristic(frames):
    N, H, W, C = frames.shape  # noqa
    # gather border using heuristic (no more than 4 pixels or 5% of total image
    #  and at least 1 pixel)
    # derivation:
    # (2 * x * H + 2 * x * (W - 2 * x)) / H * W = .05
    # 2 * x * (H + W - 2 * x) / H * W = .05
    # x * (H + W - 2 * x) = .05 * H * W / 2
    # -2 * x ** 2 + x * (H + W) - .05 * H * W / 2 = 0
    # x = (-(H + W) + ((H + W) ** 2 - 4 * 2 * .05 * H * W / 2) ** .5) / (2 * -2)
    # x = (-(H + W) + ((H + W) ** 2 - 4 * .05 * H * W) ** .5) / -4
    max_pct = 0.05  # parameter for heuristic
    n_pixels_max = (-(H + W) + ((H + W) ** 2 - 4 * max_pct * H * W) ** .5) / -4
    n_pixels_max = round(n_pixels_max)
    border_pixels = max(1, min(4, n_pixels_max))
    return border_pixels


def extract_border(frames, n_pixels):
    import numpy as np

    N, H, W, C = frames.shape  # noqa
    shape = (-1, C)
    border_top = frames[:, :n_pixels, :, :].reshape(shape)
    border_bottom = frames[:, -n_pixels:, :, :].reshape(shape)
    border_left = frames[:, n_pixels:-n_pixels, :n_pixels, :].reshape(shape)
    border_right = frames[:, n_pixels:-n_pixels, -n_pixels:, :].reshape(shape)
    border = np.concatenate(
        [border_top, border_bottom, border_left, border_right], axis=0)
    return border


def where_border_equals_color(frame, color, n_pixels):
    import numpy as np

    H, W, C = frame.shape  # noqa

    top_h, top_w, _ = np.where(frame[:n_pixels, :, :] == color)
    bottom_h, bottom_w, _ = np.where(frame[-n_pixels:, :, :] == color)
    bottom_h += H - n_pixels  # normalize
    left_h, left_w, _ = np.where(
        frame[n_pixels:-n_pixels, :n_pixels, :] == color)
    left_h += n_pixels  # normalize
    right_h, right_w, _ = np.where(
        frame[n_pixels:-n_pixels, -n_pixels:, :] == color)
    right_h += n_pixels  # normalize
    right_w += W - n_pixels  # normalize
    where_h = np.concatenate([top_h, bottom_h, left_h, right_h])
    where_w = np.concatenate([top_w, bottom_w, left_w, right_w])
    return where_h, where_w


def main():
    import argparse
    import sys

    parser = argparse.ArgumentParser(
        description='Automatically make GIF backgrounds transparent.',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        'input_filename', help='Path to the GIF'
    )
    parser.add_argument(
        'output_filename', nargs='?', default=None,
        help='Path to the output GIF (default: '
             '<input-basename>-transparent.gif)'
    )
    parser.add_argument(
        '--threshold', '-t', type=int, default=5,
        help=('The threshold for all RGB channels to determine which pixels '
              'belong to the background')
    )
    args = parser.parse_args()
    input_filename = args.input_filename
    output_filename = args.output_filename

    if not os.path.exists(input_filename):
        sys.exit(f'Error: "{input_filename}" does not exist!')

    run(
        path_gif_in=input_filename,
        path_gif_out=output_filename,
        threshold=args.threshold,
    )


if __name__ == '__main__':
    main()
