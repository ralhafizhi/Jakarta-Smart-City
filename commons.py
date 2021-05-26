from src.model import CustomMobileNetV2
import io
import torch
from torch import nn
from PIL import Image
import torchvision.transforms as transforms
import config as cfg

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# load model mnetv2


def get_model():
    checkpoint_path = torch.load(
        'model/weights_best_mnet_sampler.pth', map_location="cpu")
    model = CustomMobileNetV2(cfg.output_size).to(device)
    model.load_state_dict(checkpoint_path)
    with torch.no_grad():
        model.eval()
        return model

# data preprocessing


def get_tensor(image_location):
    my_transforms = transforms.Compose([transforms.Resize(256),
                                        transforms.CenterCrop(cfg.crop_size),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    image = Image.open(image_location)
    return my_transforms(image).unsqueeze(0)
