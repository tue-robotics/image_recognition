import numpy as np
from torchvision import transforms as T
import torch
from PIL import Image, ImageOps

def preprocess_RGB(img):
    """preproces image:
    input is a PIL image.
    Output image should be pytorch tensor that is compatible with your model"""
    img = T.functional.resize(img, size=(32, 32), interpolation=Image.NEAREST)
    trans = T.Compose([T.ToTensor(),T.Grayscale(num_output_channels=3),T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
    img = trans(img)
    img = img.unsqueeze(0)

    return img
def heroPreprocess(img):
    """preproces image:
    expected input is a PIL image from Hero.
    Output image should be pytorch tensor that is compatible with your model"""
    width, height = img.size # Hero image size (640x480)
    left = width/2 - 100
    top = height/2 + 140
    right = width/2
    bottom = height
    im1 = img.crop((left, top, right, bottom))
    img2 = T.functional.resize(im1, size=(32, 32), interpolation=Image.NEAREST)
    trans = T.Compose([T.ToTensor(),T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
    img_trans = trans(img2)
    img_trans = img_trans.unsqueeze(0)

    return img_trans

def detection_RGB(img, model):
    """Detection of foortwear:
    Input is a preprocessed image to provide to the model.
    Output should be binary classification [True, False], where True is the detection of the footwear."""
    model.eval()
    info = next(model.parameters()) # Retrieve the first parameter tensor from the iterator
    device = info.device
    dtype  = info.dtype
    with torch.no_grad():
        img    = img.to(device=device, dtype=dtype)
        scores = model(img)
        preds  = torch.argmax(scores, axis=1)
        score_max_numpy = int(preds.cpu().detach().numpy())
    return score_max_numpy
