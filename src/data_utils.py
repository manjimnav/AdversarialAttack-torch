import torch
import torch.nn.functional as F
import torchvision.transforms as TF

from PIL import Image
import json

def load_image(path):
    """
    Load and process the image to modify
    """
    min_img_size = 224 
    transform_pipeline = TF.Compose([TF.Resize(min_img_size),
                                             TF.ToTensor(),
                                             #TF.Normalize(mean=[0.485, 0.456, 0.406],
                                             #                     std=[0.229, 0.224, 0.225])
                                    ])
    image = Image.open(path)
    img_tensor = transform_pipeline(image)
    img_tensor.unsqueeze_(0)
    
    return img_tensor
