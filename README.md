`visionrt` **- Zero-overhead real-time computer vision.**

Skip the overhead:
```python
from visionrt import Camera, Preprocessor
import visionrt

camera = Camera("/dev/video0", deterministic=True)
preprocess = Preprocessor()
model = visionrt.compile(model)

for frame in camera.stream():
    tensor = preprocess(frame)
    out = model(tensor)
```

So fast you can see your camera's true refresh rate:

![kde](images/latency_histogram.png)

The orange narrow peak show `visionrt` is so fast and deterministic that you can actually see the hardware, nearly **100%** of inference runs complete at the webcam's refresh rate.

---

### Install
```bash
uv pip install git+https://github.com/abiel-almonte/visionrt
```

> Requires CUDA 12.8+, Python 3.11, PyTorch 2.8, V4L2 compatible camera.  
> Add your user to the video group: `sudo usermod -aG video $USER`
