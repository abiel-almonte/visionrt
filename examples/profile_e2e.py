import os
import copy

import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights

import nvtx

from visionrt import Camera, Preprocessor, compile
import visionrt.config as config

mean = torch.tensor([0.485, 0.456, 0.406], device="cuda").view(1, 3, 1, 1)
std = torch.tensor([0.229, 0.224, 0.225], device="cuda").view(1, 3, 1, 1)


def profile(name, fn, iters):
    with nvtx.annotate(f"profile_{name}"):
        for _ in range(iters):
            fn()


if __name__ == "__main__":
    config.verbose = True
    iters = int(os.environ.get("ITERS", "1000"))

    model = (
        nn.Sequential(
            nn.Upsample(size=(224, 224), mode="bilinear", align_corners=False),
            resnet50(weights=ResNet50_Weights.IMAGENET1K_V2),
        )
        .cuda()
        .eval()
    )

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
    cap.set(cv2.CAP_PROP_FPS, 90)
    cap.set(
        cv2.CAP_PROP_BUFFERSIZE, 1
    )  # set to one buffer so opencv cannot "cheat" - meaning, load a frame buffer before the next profiling run.
    frame = cap.read()[-1]

    @nvtx.annotate("baseline_e2e", color="blue")
    def baseline():
        _, frame = cap.read()
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        chw = np.transpose(rgb, (2, 0, 1))
        tensor = torch.from_numpy(chw).unsqueeze(0).cuda().float()
        tensor = ((tensor / 255.0) - mean) / std
        out = model(tensor)
        torch.cuda.synchronize()
        return out

    for _ in range(10):  # warmup
        _, frame = cap.read()
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        chw = np.transpose(rgb, (2, 0, 1))
        tensor = torch.from_numpy(chw).unsqueeze(0).cuda().float()
        tensor = ((tensor / 255.0) - mean) / std
        model(tensor)

    profile("baseline", baseline, iters)
    cap.release()

    cam = Camera("/dev/video0", deterministic=True)
    preprocess = Preprocessor()

    class PreprocessorWrapper(nn.Module):
        def __init__(self, preprocesser):
            super().__init__()
            self.preprocesser = preprocesser

        def forward(self, frame):
            return self.preprocesser._call_cuda(frame)

    # add inference optimizations
    config.cudagraphs = True
    config.optims.fold_conv_bn = True

    model = (
        nn.Sequential(
            PreprocessorWrapper(
                preprocess
            ),  # sequential trick, put the yuyv kernel into the model so we can include it in the cuda graph capture
            nn.Upsample(size=(224, 224), mode="bilinear", align_corners=False),
            resnet50(weights=ResNet50_Weights.IMAGENET1K_V2),
        )
        .cuda()
        .eval()
    )

    model = compile(model)

    @nvtx.annotate("visionrt_e2e", color="blue")
    def visionrt():
        out = model(next(cam))  # model includes preprocessing
        # torch.cuda.synchronize() the graph executor already does a sync on the capture stream
        return out

    for _ in range(10):
        out = model(next(cam))

    profile("visionrt", visionrt, iters)
    cam.close()
