import cv2
from PIL import Image

def get_image_details(image_path):
    # Load image using OpenCV
    img = cv2.imread(image_path)
    
    if img is None:
        print("Error: Unable to load image.")
        return
    
    # Get dimensions
    height, width, channels = img.shape
    
    # Calculate aspect ratio
    aspect_ratio = width / height
    
    # Print details
    print(f"Image Path: {image_path}")
    print(f"Width: {width} pixels")
    print(f"Height: {height} pixels")
    print(f"Aspect Ratio: {aspect_ratio:.2f}")
    print(f"Number of Channels: {channels}")
    
    # Load image using PIL to get additional details
    pil_img = Image.open(image_path)
    
    print(f"Format: {pil_img.format}")
    print(f"Mode: {pil_img.mode}")
    print(f"Size: {pil_img.size} (Width, Height)")
    
    # Check for transparency (alpha channel)
    if pil_img.mode in ('RGBA', 'LA') or (pil_img.mode == 'P' and 'transparency' in pil_img.info):
        print("Image has transparency (alpha channel).")
    else:
        print("Image does not have transparency (no alpha channel).")

# Example usage
image_path = '297.jpg'
get_image_details(image_path)
