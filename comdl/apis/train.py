#utils
from ..utils.train_utils import *

#mmcv
from mmcv import Config

##mmdet
import mmdet
from mmdet.apis import set_random_seed
from mmdet.datasets import build_dataset
from mmdet.models import build_detector
from mmdet.apis import train_detector

#other
import os
import random
from abc import ABC, abstractmethod
import torch

def get_configs():
    configs = {}
    for path, subdirs, files in os.walk(os.path.join("/".join(mmdet.__file__.split("\\")[:-1]),".mim", "configs")):
        if "_base_" in path or "__pycache__" in path:
            continue
        method = os.path.normpath(path).split(os.path.sep)[-1]
        configs[method] = []
        for name in files:
            if not name.endswith(".py"):
                continue
            configs[method].append(os.path.join(path, name))
    return configs


class BaseTrainer(ABC):
    """Base trainer class.
    """
    def __init__(self, nc, base_config, work_dir, batch_size, workers, data_root, seed, gpu_ids, load_from, save_best_on, eval_interval, epochs, custom_data):
        self.cfg = Config.fromfile(base_config)
        random.seed(random)

        # self.nc = nc
        self.cfg.seed = seed
        self.cfg.gpu_ids = gpu_ids
        self.cfg.log_config.interval = 1
        self.cfg.data.samples_per_gpu = batch_size
        self.cfg.data.workers_per_gpu = workers
        self.cfg.work_dir = work_dir
        self.cfg.load_from = load_from
        self.cfg.evaluation.interval = eval_interval
        self.cfg.evaluation.save_best=save_best_on
        self.cfg.data_root = data_root
        self.cfg.max_epochs = epochs
        self.cfg.runner.max_epochs = epochs # Epochs for the runner that runs the workflow 
        self.cfg.total_epochs = epochs


        # self.model = None
        self.custom_data = custom_data
    @abstractmethod
    def train(self):
        pass

class MMDetTrainer(BaseTrainer):
    """Responsible for training a model.
    """
    def __init__(self, nc, base_config, work_dir, batch_size, workers,
                 data_root, seed = 1234, gpu_ids = [0], load_from = None, 
                 save_best_on = 'bbox_mAP', eval_interval = 10, epochs = 100,
                 classes = None, max_per_image = 100, rpn_max_per_image = 2000,
                 img_scale = None,
                 class_agnostic = True, custom_data = {}):
        super().__init__(nc, base_config, work_dir, batch_size, workers, data_root, seed,gpu_ids, load_from, save_best_on, eval_interval, epochs, custom_data)

        #self.cfg.dump(os.path.join(self.cfg.work_dir, os.path.basename(base_config)))


        classes = tuple([str(i) for i in range(nc)]) if classes is None else classes

        #Change all to Coco
        change_key(self.cfg.data.train, "ann_file", os.path.join(data_root, "annotations/instances_train2017.json"))
        change_key(self.cfg.data.train, "img_prefix", os.path.join(data_root, "train2017") + "/")
        change_key(self.cfg.data.train, "seg_prefix", os.path.join(data_root, "train2017") + "/")

        change_key(self.cfg.data.val, "ann_file", os.path.join(data_root, "annotations/instances_val2017.json"))
        change_key(self.cfg.data.val, "img_prefix", os.path.join(data_root, "val2017") + "/")
        change_key(self.cfg.data.val, "seg_prefix", os.path.join(data_root, "val2017") + "/")

        change_key(self.cfg.data.test, "ann_file", os.path.join(data_root, "annotations/instances_val2017.json"))
        change_key(self.cfg.data.test, "img_prefix", os.path.join(data_root, "val2017") + "/")
        change_key(self.cfg.data.val, "seg_prefix", os.path.join(data_root, "val2017") + "/")

        # #TODO: This is because MultiImageDataset
        if "dataset" in self.cfg.data.train:
            self.cfg.data.train.dataset.classes = classes
        else:
            self.cfg.data.train.classes = classes
        self.cfg.data.val.classes = classes
        self.cfg.data.test.classes = classes
        

        change_key(self.cfg, "dataset_type", "CocoDataset")
        change_key(self.cfg, "persistent_workers", False)
        if class_agnostic:
            try:
                add_key(self.cfg.model.test_cfg.rcnn, "iou_threshold", "class_agnostic", True)
            except:
                pass
        self.cfg.evaluation['save_best'] = 'bbox_mAP'
        
        if img_scale is not None:
            change_key(self.cfg, "img_scale", img_scale)

        #change_key(self.cfg.model, "score_thr", 0.05)

        change_keys(self.cfg,[("num_classes", nc, ""),
                  ("max_per_img", max_per_image, ""),
                  ("max_per_img", rpn_max_per_image, "nms_pre"),
                  ("nms_pre", rpn_max_per_image, ""),
                  ("classes", classes, "")])

        # change_key(self.cfg.model.train_cfg, "nms_pre", rpn_max_per_image + 1000)

    

    def train(self, validate = True):
        self.model = build_detector(self.cfg.model,
                                    train_cfg=self.cfg.get("train_cfg"),
                                    test_cfg=self.cfg.get('test_cfg'))
        self.cfg.dump(os.path.join(self.cfg.work_dir, "config.py"))
        self.model.init_weights()
        datasets = [build_dataset(self.cfg.data.train)]
        train_detector(self.model, datasets[0], self.cfg, distributed=False, validate=validate)

        #find best_checkpoint
        bests = []
        if self.cfg.work_dir == "":
            for file in os.listdir():
                if("best" in file):
                    bests.append(file)
        else:
            for file in os.listdir(self.cfg.work_dir):
                if("best" in file):
                    bests.append(file)
        bests = sorted(bests, key = lambda x:int(x.split("epoch_")[-1].replace(".pth", "")))
        data = torch.load(bests[-1], map_location=torch.device("cpu"))
        data['custom_data'] = self.custom_data
        torch.save(os.path.join(self.cfg.work_dir, "best_model.pt"))



    

