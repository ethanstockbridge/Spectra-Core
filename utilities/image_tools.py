import os

from PIL import Image


def compress_image(input_path, output_path, target_size_kb=2, quality=85):
    """Image compression class

    Args:
        input_path (str): input image
        output_path (str): output location for image
        target_size_kb (int, optional): Target size. Defaults to 2.
        quality (int, optional): Quality. Defaults to 85.
    """
    try:
        img = Image.open(input_path)

        # Adjust the quality to achieve the target size
        while True:
            with open(output_path, 'wb') as output_file:
                img.save(output_file, format='JPEG', quality=quality)
            
            compressed_size_kb = os.path.getsize(output_path) / 1024.0

            if compressed_size_kb <= target_size_kb:
                print(compressed_size_kb, target_size_kb)
                print(compressed_size_kb <= target_size_kb)
                print(f"Compressed successfully to {compressed_size_kb}kb")
                break

            # Decrease the quality for further compression
            quality -= 5
            if quality < 5:
                # If quality drops too low, break the loop
                print(f"Quality dropped too far, quitting at lowest: {compressed_size_kb}")
                break

        print(f"Image compressed successfully to {compressed_size_kb:.2f} KB")

    except Exception as e:
        print(f"Error compressing image: {e}")