import os
import torch
import numpy as np

import cv2
from visionrt import Camera, Preprocessor
import visionrt.config as config
import nvtx

_preprocess = Preprocessor()
mean = torch.tensor([0.485, 0.456, 0.406], device="cuda").view(1, 3, 1, 1)
std = torch.tensor([0.229, 0.224, 0.225], device="cuda").view(1, 3, 1, 1)


@nvtx.annotate("opencv_camera", color="blue")
def opencv_camera(cap):
    _, frame = cap.read()
    return frame


@nvtx.annotate("visionrt_camera", color="red")
def visionrt_camera(cap):
    return next(cap)


@nvtx.annotate("opencv_preprocess", color="blue")
def opencv_preprocess(frame):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    chw = np.transpose(rgb, (2, 0, 1))
    tensor = torch.from_numpy(chw).unsqueeze(0).cuda().float()
    tensor = ((tensor / 255.0) - mean) / std
    return tensor


@nvtx.annotate("visionrt_preprocess", color="red")
def visionrt_preprocess(frame):
    return _preprocess._call_cuda(frame)


@nvtx.annotate("opencv", color="blue")
def opencv(cap):
    _, frame = cap.read()
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    chw = np.transpose(rgb, (2, 0, 1))
    tensor = torch.from_numpy(chw).unsqueeze(0).cuda().float()
    tensor = ((tensor / 255.0) - mean) / std
    return tensor


@nvtx.annotate("visionrt", color="red")
def visionrt(cam):
    frame = next(cam)
    return _preprocess._call_cuda(frame)


def profile(name, fn, inp, iters):
    with nvtx.annotate(f"profile_{name}"):
        for _ in range(iters):
            fn(inp)


if __name__ == "__main__":
    config.verbose = True
    iters = int(os.environ.get("ITERS", "1000"))

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
    cap.set(cv2.CAP_PROP_FPS, 90)
    frame = cap.read()[-1]
    profile("opencv_capture", opencv_camera, cap, iters)

    for _ in range(10): # warmup
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        chw = np.transpose(rgb, (2, 0, 1))
        tensor = torch.from_numpy(chw).unsqueeze(0).cuda().float()
        tensor = ((tensor / 255.0) - mean) / std

    profile("opencv_preprocess", opencv_preprocess, frame, iters)
    profile("opencv", opencv, cap, iters)
    cap.release()

    cam = Camera("/dev/video0", deterministic=True)
    frame = next(cam)
    profile("visionrt_capture", visionrt_camera, cam, iters)

    for _ in range(10):
        _preprocess._call_cuda(frame) # warmup

    profile("visionrt_preprocess", visionrt_preprocess, frame, iters)
    profile("visionrt", visionrt, cam, iters)
    cam.close()
