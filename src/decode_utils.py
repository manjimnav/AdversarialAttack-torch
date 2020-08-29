
import torch
import numpy as np
import json

def read_idx_2_label():
    """
    Read the dictionary that converts the model output index to the imagenet class label
    Source: https://gist.github.com/yrevar/942d3a0ac09ec9e5eb3a
    """
    with open('../Data/imagenet_class_index.json') as f:
        dictionary = json.load(f)
    return dictionary

def decode(model, image, dictionary):
    """
    Returns the probability, the index and the label of the predicted class
    """

    prediction = model(image)
    sm = torch.nn.Softmax(dim=1)
    probabilities = sm(prediction) 

    idx = torch.argmax(probabilities)
    label = dictionary[str(idx.item())]
    prob = np.round(probabilities[0][idx].item(), 2)
    
    return prob, idx, label
