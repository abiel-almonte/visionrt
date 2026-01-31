import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights

import visionrt
from visionrt import Camera, Preprocessor


visionrt.config.verbose = True
visionrt.config.optims.fold_conv_bn = True
visionrt.config.cudagraphs = True


model = (
    nn.Sequential(
        Preprocessor(use_triton=False),  # cuda is faster
        nn.Upsample(size=(224, 224), mode="bilinear", align_corners=False),
        resnet50(weights=ResNet50_Weights.IMAGENET1K_V2),
        nn.Softmax(dim=1),
    )
    .cuda()
    .eval()
)
labels = ResNet50_Weights.IMAGENET1K_V2.meta["categories"]

camera = Camera("/dev/video0", deterministic=True)
optimized_model = visionrt.compile(model)

print(optimized_model)  # snapshot of the config

with torch.inference_mode():
    for frame in camera.stream():
        out = optimized_model(frame)

        pred_class = out.argmax(dim=1).item()
        label = labels[pred_class]
        print(f"Predicted class: {label}")

print(optimized_model)  # cuda graph status should change
