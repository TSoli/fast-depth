import argparse
import time

import cv2
import numpy as np
import torch
from torchvision import transforms


def pad(img: cv2.typing.MatLike, size: tuple[int, ...]) -> cv2.typing.MatLike:
    """
    Pad an image to size.
    """
    h, w, c = img.shape
    top = (size[0] - h) // 2
    bottom = size[0] - h - top
    left = (size[1] - w) // 2
    right = size[1] - w - left
    return cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT)


def crop_middle(img: cv2.typing.MatLike, size) -> cv2.typing.MatLike:
    """
    Crop an image to a square. Assuming width is larger.
    """
    h, w, c = img.shape
    top = (h - size[0]) // 2
    bottom = h - top
    left = (w - size[1]) // 2
    right = w - left
    return img[top:bottom, left:right, :]


def preprocess_frame(frame: cv2.typing.MatLike) -> torch.Tensor:
    """
    preprocess frame for the model.
    """
    long_idx = np.argmax(frame.shape)
    short_idx = (long_idx + 1) % 2
    side_length = frame.shape[short_idx]
    frame = crop_middle(frame, (side_length, side_length))
    frame = cv2.resize(frame, (224, 224))
    frame = frame.astype(np.float32) / 255
    input_tensor = transforms.ToTensor()(frame)
    # for batch dim
    return input_tensor.unsqueeze(0).cuda()


def main(args):
    ckpt = torch.load(args.ckpt)
    model = ckpt["model"]
    cap = cv2.VideoCapture(args.video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    last_time = None
    skip_frames = 10
    curr_frame = -1

    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            break

        curr_frame += 1
        # if curr_frame % 5:
        #     continue

        # depth = model(preprocess_frame(frame))
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Resize((224, 224))]
        )
        depth = model(transform(frame_rgb).unsqueeze(0).cuda())
        depth = depth.squeeze()
        # max 4m
        depth = 255.0 * depth / 4
        depth = torch.clamp(depth, 0, 255.0)
        depth_img = depth.detach().cpu().numpy().astype(np.uint8)
        depth_img = cv2.resize(depth_img, (960, 720))
        depth_img = cv2.applyColorMap(depth_img, cv2.COLORMAP_INFERNO)

        # long_idx = np.argmax(frame.shape)
        # short_idx = (long_idx + 1) % 2
        # side_length = frame.shape[short_idx]

        # frame_img = crop_middle(frame, (side_length, side_length))
        # frame_img = cv2.resize(frame_img, (224, 224))
        video_img = np.concatenate((frame, depth_img), axis=1)
        if last_time is None:
            last_time = time.time()

        cv2.imshow("video", video_img)

        curr_time = time.time()
        wait_time = round(1000 * ((1 / fps) - (curr_time - last_time)))
        if cv2.waitKey(wait_time if wait_time > 0 else 1) & 0xFF == ord("q"):
            break
        # key = cv2.waitKey(10)
        # while key == -1:
        #     key = cv2.waitKey(10)
        #
        # if key == ord("q"):
        #     break

        last_time = time.time()

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("ckpt", help="Checkpoint file for model")
    parser.add_argument("video", help="video file to process.")
    main(parser.parse_args())
