import copy
import random
import torch
import torchvision.transforms as transforms
from skimage.util import random_noise


def datapoison(dataset, indices, poisoned_persent, poisoned_ratio, targeted=False, target_label=3, noise_type="salt"):
    """
    Function to add random noise of various types to the dataset.
    """
    new_dataset = copy.deepcopy(dataset)
    train_data = new_dataset.data
    targets = new_dataset.targets
    num_indices = len(indices)
    print(target_label)

    if targeted == False:
        num_poisoned = int(poisoned_persent*num_indices)
        if num_indices == 0:
            return new_dataset
        if num_poisoned > num_indices:
            return new_dataset
        poisoned_indice = random.sample(indices, num_poisoned)
        
        for i in poisoned_indice:
            t = train_data[i]
            if noise_type == "salt":
                # Replaces random pixels with 1.
                poisoned = torch.tensor(random_noise(t, mode=noise_type, amount=poisoned_ratio))
            elif noise_type == "gaussian":
                # Gaussian-distributed additive noise.
                poisoned = torch.tensor(random_noise(t, mode=noise_type, mean=0, var=poisoned_ratio, clip=True))
            elif noise_type == "s&p":
                # Replaces random pixels with either 1 or low_val, where low_val is 0 for unsigned images or -1 for signed images.
                poisoned = torch.tensor(random_noise(t, mode=noise_type, amount=poisoned_ratio))
            else:
                print("ERROR: poison attack type not supported.")
                poisoned = t
            train_data[i] = poisoned 
    else:
        for i in indices:            
            if int(targets[i]) == int(target_label):
                t = train_data[i]
                poisoned = add_x_to_image(t)
                train_data[i] = poisoned
    new_dataset.data = train_data
    return new_dataset

def add_x_to_image(img):
    """
    Add a 10*10 pixels X at the top-left of a image
    """
    for i in range(0,10):
        for j in range(0,10):
            if i+j<=9 or i==j:
                img[i][j]=255
    return torch.tensor(img)