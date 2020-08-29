import matplotlib.pyplot as plt
from decode_utils import decode


def plot_images(image_modified, image, model, dictionary):
    prob_mod, idx_mod, label_mod = decode(model, image_modified, dictionary)
    prob_or, idx_or, label_or = decode(model, image, dictionary)

    image_modified = (image_modified.squeeze().permute(1, 2, 0).detach().numpy()*255).astype(int)
    original_modified = (image.squeeze().permute(1, 2, 0).detach().numpy()*255).astype(int)
    
    fig, ax = plt.subplots(1,2,figsize=(20, 20))
    ax[0].imshow(image_modified)
    ax[0].set_title(f'Modified ({label_mod[1]})')
    ax[1].imshow(original_modified)
    ax[1].set_title(f'Origial ({label_or[1]})')

def show_predictions(*args):
    args = args[0] 
    print(f"Predicted as {args[2][1]} at index {args[1]} with a probability of {args[0]}%")
