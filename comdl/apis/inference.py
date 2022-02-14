import numpy as np
from mmdet.apis.inference import init_detector, inference_detector
#TODO: from mmseg.apis.inference import ...

class InferenceModel:
    """A wrapper around the mmdet inference api
    """
    def __init__(self, config, checkpoint, device = "cuda:0", cfg_options=None):
        """Initializes the InferenceModel

        Args:
            config ([type]): The config file.
            checkpoint ([type]): The path to the checkpoint.
            device (str, optional): [description]. Defaults to "cuda:0".
            cfg_options ([type], optional): [description]. Defaults to None.
        """
        #TODO: Implement mmseg init
        self.model = init_detector(config, checkpoint, device, cfg_options)


    def __call__(self, x):
        """Forwards x to the inference_XXX function

        Args:
            x (np.array): [description]
        """
        return inference_detector(self.model, x)