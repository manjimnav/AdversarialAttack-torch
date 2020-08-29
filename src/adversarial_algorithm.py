import torch
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

def modify_image(image, model, target_class, max_allowed_change=0.01, min_prob=1, max_epochs=300):
    """
    Update the image until the target class is the desired.
    """
    
    target_class = torch.tensor([target_class])
    predicted_class = -1
    prob = -1
    
    # Calculate the maximum and minimum change images
    max_change_above = image + max_allowed_change
    max_change_below = image - max_allowed_change

    hacked_image = Variable(image.clone(), requires_grad=True)
    
    max_epochs = max_epochs
    i = 0

    while predicted_class != target_class or (predicted_class == target_class and prob<min_prob):

        # Obtain the predicted class
        output = model(hacked_image)
        sm = torch.nn.Softmax(dim=1)
        probabilities = sm(output)
        idx = torch.argmax(probabilities) 
        prob = probabilities[0][idx]

        predicted_class = output.max(1, keepdim=True)[1]

        #Calculate the loss
        loss = F.nll_loss(output, target_class)

        # Zero all existing gradients
        model.zero_grad()

        # Calculate gradients of model in backward pass
        loss.backward(retain_graph=True)

        # Collect datagrad
        data_grad = hacked_image.grad.data

        # Modify the image descending in the gradient
        hacked_image = hacked_image - 0.1*data_grad

        # Workaround to clip the image for tensors
        hacked_image = torch.where(hacked_image > max_change_above, max_change_above, hacked_image)
        hacked_image  = torch.where(hacked_image < max_change_below, max_change_below, hacked_image)
        hacked_image.retain_grad()

        if i % 10==0 or predicted_class == target_class:
            print(f"Iteration: {i}, Prob: {np.round(probabilities[0][idx].item(), 2)}, actual class is: {predicted_class.squeeze()}")
        if i==max_epochs:
            print("Max epochs reached, aborting")
            break

        i+=1
        
    #hacked_image = (hacked_image.squeeze().permute(1, 2, 0).detach().numpy()*255).astype(int)
    return hacked_image

