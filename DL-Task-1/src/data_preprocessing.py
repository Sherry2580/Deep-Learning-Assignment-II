import os
import cv2
import numpy as np

def preprocess_image(image_path, output_path, size=(256, 256)):
    """
    Preprocess the image, resize it to a uniform size, and ensure it has three channels (RGB).

    Parameters:
    - image_path: Path to the input image
    - output_path: Path to the output image
    - size: Resized image size, default is (256, 256)
    """
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    
    if image is None:
        print(f"Warning: Unable to read image {image_path}. Skipping...")
        return
    
    if len(image.shape) == 2:  # 單通道
        image = cv2.merge([image, np.zeros_like(image), np.zeros_like(image)])
    elif image.shape[2] == 2:  # 雙通道 空缺的通道設為0
        empty_channel = np.zeros_like(image[:, :, 0])
        image = cv2.merge([image[:, :, 0], image[:, :, 1], empty_channel])
    elif image.shape[2] > 3:  # 多於三個通道
        image = image[:, :, :3]  # 僅保留前三個通道
    
    image = cv2.resize(image, size) 
    cv2.imwrite(output_path, image)  

def preprocess_data(txt_file, input_folder, output_folder, size=(256, 256)):
    """
    Batch preprocess image data, resize them to a uniform size, and ensure they have three channels (RGB).

    Parameters:
    - txt_file: Text file containing image paths and labels
    - input_folder: Root directory of the input images
    - output_folder: Root directory of the output images
    - size: Resized image size, default is (256, 256)
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    with open(txt_file, 'r') as file:
        lines = file.readlines()
        total_lines = len(lines)
        
        for i, line in enumerate(lines):
            relative_path, label = line.strip().split()
            input_path = os.path.join(input_folder, relative_path)
            output_path = os.path.join(output_folder, relative_path)
            
            output_dir = os.path.dirname(output_path)
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            
            preprocess_image(input_path, output_path, size)
            
            if (i + 1) % 100 == 0 or (i + 1) == total_lines:
                print(f"Processed {i + 1}/{total_lines} images.")

    print("Done!")

if __name__ == "__main__":
    input_folder = 'data'
    output_folder = 'data/processed'
    txt_files = ['data/train.txt', 'data/val.txt', 'data/test.txt']
    
    for txt_file in txt_files:
        preprocess_data(txt_file, input_folder, output_folder)
