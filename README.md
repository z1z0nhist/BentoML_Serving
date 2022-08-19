# BentoML_Serving
BentoML Serving with yolov5(torch)

# 1. init WrapperModel then save model as bentoml model

```
import torch
from models.common import DetectMultiBackend
from utils.torch_utils import select_device, time_sync
from pathlib import Path
from utils.datasets import IMG_FORMATS, VID_FORMATS, LoadImages
from utils.general import check_file, check_img_size, non_max_suppression, scale_coords
import bentoml

# Model
device = select_device('')
original_model = DetectMultiBackend('best.pt', device=device, dnn=False)


class WrapperModel(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, imgs):
        source = str(imgs)
        is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
        is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
        if is_url and is_file:
            source = check_file(source)  # download

        # Load model
        device = select_device('')
        stride, names, pt, jit, onnx, engine = self.model.stride, self.model.names, self.model.pt, self.model.jit, self.model.onnx, self.model.engine
        imgsz = check_img_size((448, 448), s=stride)  # check image size

        half = False
        if pt or jit:
            self.model.model.half() if half else model.model.float()

        # Dataloader
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt)
        bs = 1  # batch_size
        vid_path, vid_writer = [None] * bs, [None] * bs

        # Run inference
        dt, seen = [0.0, 0.0, 0.0], 0
        for path, im, im0s, vid_cap, s in dataset:
            t1 = time_sync()
            im = torch.from_numpy(im).to(device)
            im = im.half() if half else im.float()  # uint8 to fp16/32
            im /= 255  # 0 - 255 to 0.0 - 1.0
            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim
            t2 = time_sync()
            dt[0] += t2 - t1

            # Inference
            pred = self.model(im)
            t3 = time_sync()
            dt[1] += t3 - t2

            # NMS
            pred = non_max_suppression(prediction=pred, conf_thres=torch.tensor(0.25).cuda(),
                                       iou_thres=torch.tensor(0.45).cuda(), classes=None, agnostic=False, max_det=1000)
            dt[2] += time_sync() - t3

            # Process predictions
            for i, det in enumerate(pred):  # per image
                seen += 1
                p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)

                s += '%gx%g ' % im.shape[2:]  # print string
                if len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()

                    # Print results
                    for c in det[:, -1].unique():
                        n = (det[:, -1] == c).sum()  # detections per class
                        s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string
        return s


model = WrapperModel(original_model)
```

# 2. write service.py

```
import bentoml
from bentoml.io import Text

yolo_runner = bentoml.pytorch.get("pytorch_yolov5").to_runner()

svc = bentoml.Service(
    name="pytorch_yolo_demo",
    runners=[yolo_runner],
)


@svc.api(input=Text(), output=Text())
async def predict(img: str) -> str:
    assert isinstance(img, str)
    return await yolo_runner.async_run(img)
```

# 3. Run service

```
bentoml serve service.py:svc
```

# 4. Send API

```
curl -X POST -H "Content-Type: text/plain" --data 'SAMPLE IMG URI' http://localhost:3000/predict
```

return 'image 1/1 /home/halo/PycharmProjects/bentoml/202106180043369591_0.jpg: 352x448 1 F_Duro_50mcg,'

# Options

```
# Production
bentoml serve --production --host 0.0.0.0 service.py:svc
```
