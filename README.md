# InferenceDL

A repository for MaskRCNN and Vision Transformer implementations in HSA KIT 1.5.x. 

> Note: This repository is largely deprecated.

## Installation

### Option 1: HSA KIT Environment (Recommended)
The easiest way to install is to use the 1.5.x HSA KIT environment.

### Option 2: Standalone Installation
```bash
# Install PyTorch dependencies
pip install torch==1.10.1+cu113 torchvision==0.11.2+cu113 torchaudio===0.10.1+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html

# Install MMCV
pip install mmcv-full==1.6.0 -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.10.0/index.html

# Install MMDetection and InferenceDL
pip install git+https://github.com/hs-analysis/mmdetection.git
pip install git+https://github.com/hs-analysis/InferenceDL.git
```

## Usage

### Training

The training process uses the MMDetection framework as a backend. Here's an example of how to start training:

```python
from comdl.apis.train import MMDetTrainer, get_configs
import numpy as np
import torch

# Choose model configuration
# For Vision Transformer (Swin):
# base_config = get_configs()['swin'][3]  # "mask_rcnn_swin-s-p4-w7_fpn_fp16_ms-crop-3x_coco"
# For Faster R-CNN:
base_config = list(filter(lambda x: "r50_fpn_1x_coco" in x, get_configs()["faster_rcnn"]))[0]

# Configure pretrained weights
# For Vision Transformer:
# load_from = "https://download.openmmlab.com/mmdetection/v2.0/swin/mask_rcnn_swin-t-p4-w7_fpn_fp16_ms-crop-3x_coco/mask_rcnn_swin-t-p4-w7_fpn_fp16_ms-crop-3x_coco_20210908_165006-90a4008c.pth"
# For Faster R-CNN:
load_from = "https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth"

# Training Configuration
trainer = MMDetTrainer(
    nc=nc,                                 # Number of classes
    base_config=base_config,               # Model configuration
    work_dir="path/to/output",            # Output directory
    batch_size=2,                          # Batch size
    workers=0,                             # Number of workers
    dataroot="path/to/data",              # Data directory
    max_per_image=100,                     # Max detections per image
    rpn_max_per_image=2000,                # Max RPN proposals
    load_from=load_from,                   # Pretrained weights
    epochs=300,                            # Number of epochs
    img_scale=(800, 1333),                 # Image scale
    eval_interval=1,                       # Evaluation frequency
    class_agnostic=False,                  # Class agnostic mode
    custom_data={...}                      # Custom data configuration
)

# Configure learning rate
trainer.cfg.lr_config = dict(
    policy='CosineAnnealing',
    by_epoch=False,
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    min_lr=1e-07
)

# Set device
trainer.cfg.device = 'cuda'

# Start training
trainer.train()
```

### Inference
First load the model:
```python
from comdl.apis.inference import InferenceModel
config_file = r"F:\source\repos\InferenceDL\Exp\config.py"
checkpoint_file = r"F:\source\repos\InferenceDL\Exp\best_bbox_mAP_epoch_36.pth"
model = InferenceModel(config_file, checkpoint_file)
```
Follow the implementation in HSA KIT:

#### Object Detection
```python
def predict_mmdet_od(self, img: np.ndarray) -> list:
    """
    Analyze image with od model of comdl module

    Args:
        img_orig (np.ndarray): image to analyze

    Returns:
        list: layers with annotations
    """

    class_preds = []
    for idx in self.iam_structure_indices:
        class_preds.append({"idx": idx, "contours": [], "pred_dicts": []})

    result = self.od_model(img)
    for i in range(len(result)):
        for bboxes in result[i]:
            # if confidence too low remove
            score = bboxes[4].astype(float)
            if score < self.confidence_threshold:
                continue

            x = 0
            y = 0
            x_end = img.shape[1]
            y_end = img.shape[0]

            bbox = bboxes[:4].astype(int)
            left = x + bbox[0]
            top = y + bbox[1]
            right = x + bbox[2]
            bottom = y + bbox[3]
            w = right - left
            h = bottom - top

            # throw away if near border and smaller than overlap (probably not full object)
            if w < self.overlap:
                if left - x < 1 or right > x_end - 1:
                    continue
            if h < self.overlap:
                if top - y < 1 or bottom > y_end - 1:
                    continue

            p_1 = [[left, top]]  # top left
            p_2 = [[right, top]]  # top right
            p_3 = [[right, bottom]]  # bottom right
            p_4 = [[left, bottom]]  # bottom left

            cnt = np.array([p_1, p_2, p_3, p_4], dtype=np.int32)
            bounds = [left, top, right, bottom]
            pred = [1, 2, 3, 4]
            pred[0] = left
            pred[1] = top
            pred[2] = right
            pred[3] = bottom
            pred_dict = {
                "idx": i,
                "pred": pred,
                "bounds": bounds,
                "cnt": cnt,
                "area": w * h,
                "confidence": score,
            }

            pred_class = self.iam_structure_indices[i]
            list_index = next(
                (
                    i
                    for i, item in enumerate(class_preds)
                    if item["idx"] == pred_class
                ),
                -1,
            )
            class_preds[list_index]["pred_dicts"].append(pred_dict)

    layers = []
    for structure in class_preds:
        layers.append(
            {
                "id": (
                    structure["idx"] if len(self.iam_structure_indices) > 1 else -1
                ),
                "contours": [],
                "pred_dicts": structure["pred_dicts"],
            }
        )

    return layers
```

#### Instance Segmentation
```python
def predict_mmdet_instance_seg(self, img: np.ndarray) -> np.ndarray:
    """
    Analyze image with instance seg model of comdl module

    Args:
        img (np.ndarray): image to analyze

    Returns:
        np.ndarray: mask with segmentation predictions of model
    """

    result = self.instance_seg_model(img)

    classes = list(range(1, len(self.iam_structure_indices) + 1))
    candidates = []

    for i in range(len(result[0])):
        for bboxes, seg in zip(result[0][i], result[1][i]):
            bbox = bboxes[:4].astype(int)
            score = bboxes[4].astype(float)
            if score < self.confidence_threshold:
                continue
            pts = np.where(seg.astype(float) > 0)
            candidates.append((score, bbox, pts, i))

    # sort by object score
    candidates.sort(key=lambda x: x[0], reverse=True)

    img_shape = (img.shape[0], img.shape[1])
    final = np.zeros(img_shape, dtype="uint8")
    contour_mask = np.zeros(img_shape, dtype="uint8")
    empty = np.zeros((img.shape[0], img.shape[1], 1))
    for cand in candidates:
        score = cand[0]
        bbox = cand[1]
        pts = cand[2]
        cls = cand[3]

        mas = np.zeros((img.shape[0], img.shape[1]))
        for y, x in zip(pts[0], pts[1]):
            mas[y, x] = 255

        # fix contour in contour
        if np.count_nonzero(final[mas == 255]) <= 100:
            final[mas == 255] = random.randint(50, 255)

            single_mask = mas.astype("uint8")
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            single_mask = cv2.dilate(single_mask, kernel, iterations=1)

            # create image with all contours
            single_contours = cv2.findContours(
                single_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
            )[0]
            contours = []
            # remove contours that are next to border and completely inside the overlap
            for cnt in single_contours:
                x, y, w, h = cv2.boundingRect(cnt)
                if (x < 2 or x + w > single_mask.shape[1] - 2) and w < self.overlap:
                    continue
                if (y < 2 or y + h > single_mask.shape[0] - 2) and h < self.overlap:
                    continue
                contours.append(cnt)

            cv2.drawContours(contour_mask, contours, -1, classes[cls], -1)
            cv2.drawContours(contour_mask, contours, -1, 0, 1)
            cv2.drawContours(
                contour_mask, contours, -1, 0, 1, cv2.LINE_4
            )  # 2nd time needed to prevent pixel issue

    return contour_mask
```

## Backend

The backend utilizes the MMDetection framework, similar to MMWrapper, but in a more rudimentary implementation.
