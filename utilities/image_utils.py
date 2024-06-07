import os

import numpy as np
from PIL import Image

from config.config_manager import ConfigManager


def spectro_gen(path, Sxx, yolo_train_size, image_gain):
    """Generate a spectrogram given the path and image data

    Args:
        path (str): Path to write the spectrogram from
        Sxx (numpy.array): Sxx
        yolo_train_size (int): Output image size (match yolo)
        image_gain (int): Amount to increase image by (-1=auto)

    Returns:
        PIL.Image: Image result
    """
    config = ConfigManager()
    # Sxx=(Sxx-np.min(Sxx))/(np.max(Sxx)-np.min(Sxx))*255 #normalize
    Sxx *= image_gain * config["spectrogram"]["default_image_gain"]
    format = config["spectrogram"]["format"]

    # Calculate RGB values for the entire image
    r_values = np.zeros_like(Sxx, dtype=np.uint8)
    g_values = np.zeros_like(Sxx, dtype=np.uint8)
    b_values = np.zeros_like(Sxx, dtype=np.uint8)
    for y in range(len(Sxx)):
        # RGB values are represented by image height
        r_values[y], g_values[y], b_values[y] = rgb(0, len(Sxx), y)

    # Apply opacity to strength
    if format == "png": #NOTE:Untested
        val = (Sxx / 255).astype(np.uint8)
        alpha_channel = val  # Using Sxx as alpha channel
        rgba_image = np.stack((r_values, g_values, b_values, alpha_channel), axis=-1)
    if format == "jpg":
        val = (Sxx / 255).astype(np.float32)
        rgba_image = np.stack((r_values * val, g_values * val, b_values * val), axis = -1)

    # Create PIL Image from RGBA array
    rgba_image_int = np.round(rgba_image).astype(np.uint8)
    img = Image.fromarray(rgba_image_int)
    img = img.transpose(Image.FLIP_TOP_BOTTOM)
    img = img.resize((yolo_train_size,yolo_train_size), Image.LANCZOS)

    if path != None:
        if not os.path.exists(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path))

        img.save(path)

    return img

def compress_image(input_path, output_path, target_size_kb=None, quality=85):
    """Image compression class

    Args:
        input_path (str): input image
        output_path (str): output location for image
        target_size_kb (int, optional): Target size. Defaults to 2.
        quality (int, optional): Quality. Defaults to None (only performs quality compression).
    """
    try:
        img = Image.open(input_path)

        # Adjust the quality to achieve the target size
        while True:
            with open(output_path, 'wb') as output_file:
                img.save(output_file, format='JPEG', quality=quality)
            
            compressed_size_kb = os.path.getsize(output_path) / 1024.0

            if target_size_kb == None:
                break
            if compressed_size_kb <= target_size_kb:
                print(compressed_size_kb, target_size_kb)
                print(compressed_size_kb <= target_size_kb)
                break

            # Decrease the quality for further compression
            quality -= 5
            if quality < 5:
                # If quality drops too low, break the loop
                print(f"Quality dropped too far, quitting at lowest: {compressed_size_kb}")
                break

        print(f"Image compressed successfully to {compressed_size_kb:.2f} KB  with quality {quality}")

    except Exception as e:
        print(f"Error compressing image: {e}")

def rgb(minimum, maximum, value):
    """Report RGB representation of a value, useful if you want to
    produce a unique color from 0-100, if value is 44, it will be a color

    Args:
        minimum (int): min range
        maximum (int): max range
        value (int): value between

    Returns:
        (int,int,int): rgb representation
    """
    minimum, maximum = float(minimum), float(maximum)
    ratio = 2*(value-minimum) / (maximum-minimum)
    b = max(0,255*(1-ratio))
    b=int(b)
    r = max(0,255*(ratio-1))
    r=int(r)
    g = 255-b-r
    return r,g,b