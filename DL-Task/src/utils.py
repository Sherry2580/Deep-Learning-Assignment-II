import os
import torch

def create_directories(dir_list):
    """
    Create a list of specified directories.

    Parameters:
    - dir_list: List of directory paths to create
    """
    for directory in dir_list:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"Created directory: {directory}")

def save_checkpoint(state, filename="checkpoint.pth.tar"):
    """
    Save a checkpoint of the model.

    Parameters:
    - state: Dictionary containing the model state
    - filename: Name of the checkpoint file
    """ 
    torch.save(state, filename)
    print(f"Checkpoint saved to {filename}")

def load_checkpoint(model, optimizer, filename="checkpoint.pth.tar"):
    """
    Load a checkpoint of the model.

    Parameters:
    - model: The model object
    - optimizer: The optimizer object
    - filename: Name of the checkpoint file
    """ 
    if os.path.isfile(filename):
        print(f"Loading checkpoint '{filename}'")
        checkpoint = torch.load(filename)
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        print(f"Checkpoint loaded from '{filename}'")
    else:
        print(f"No checkpoint found at '{filename}'")
