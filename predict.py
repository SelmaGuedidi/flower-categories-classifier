import argparse
import json

ap = argparse.ArgumentParser()

ap.add_argument('image_path', help='Path to image')
ap.add_argument('--checkpoint', help='Given checkpoint of a network')
ap.add_argument('--top_k', help=' k most likely classes')
ap.add_argument('--category_names', help='categories to real names')
ap.add_argument('--gpu', help='Use GPU for inference')
p = ap.parse_args()

image_path = p.image_path
checkpoint = p.checkpoint
top_k = p.top_k

hardware = "gpu" if p.gpu else "cpu"

import torch
import numpy as np
from torch import nn, optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from collections import OrderedDict
from PIL import Image
import seaborn as sns
import matplotlib.pyplot as plt
def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    model = models.vgg16(pretrained = True)
    for param in model.parameters():
        param.requires_grad = False
    
    model.classifier = nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(checkpoint['input_size1'], checkpoint['output_size1'])),
                          ('relu', nn.ReLU()),
                          ('dropout', nn.Dropout(checkpoint['dropout'])),
                          ('fc2', nn.Linear(checkpoint['output_size1'], checkpoint['output_size2'])),
                          ('relu', nn.ReLU()),
                          ('fc3', nn.Linear(checkpoint['output_size2'], checkpoint['output_size3'])),
                          ('output', nn.LogSoftmax(dim=1))
                          ]))


    model.class_to_idx = checkpoint['class_to_idx'] 
    epochs = checkpoint['epochs']
    learning_rate = checkpoint['learning_rate']
    return model
model=load_checkpoint('checkpoint.pth')

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    
    # TODO: Process a PIL image for use in a PyTorch model
    pil_image=Image.open(image)
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
    ])
    return transform(pil_image)
def imshow(image, ax=None, title=None):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()
    
    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.numpy().transpose((1, 2, 0))
    
    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    
    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)
    
    ax.imshow(image)
    
    return ax
def predict(image_path, model, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    
    # TODO: Implement the code to predict the class from an image file
    img_torch = process_image(image_path)
    img_torch = img_torch.unsqueeze_(0).float()
    
    with torch.no_grad():
        output = model.forward(img_torch.cuda())    
    probability = F.softmax(output.data,dim=1)
    
    return probability.topk(topk)
print(predict(image_path,model))
