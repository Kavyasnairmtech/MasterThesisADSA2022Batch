import os
import sys
import time

import ailia
import cv2
import numpy as np
import matplotlib.pyplot as plt  # For plotting

# import original modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'util')))
from logging import getLogger  # logger
from arg_utils import get_base_parser, get_savepath, update_parser  # noqa: E402
from image_utils import imread, normalize_image  # noqa: E402
from model_utils import check_and_download_models  # noqa: E402
from webcamera_utils import calc_adjust_fsize  # noqa: E402
from webcamera_utils import get_capture, get_writer

logger = getLogger(__name__)

# ======================
# Parameters
# ======================
WEIGHT_v20_PATH = 'midas.onnx'
MODEL_v20_PATH = 'midas.onnx.prototxt'
WEIGHT_v21_PATH = 'midas_v2.1.onnx'
MODEL_v21_PATH = 'midas_v2.1.onnx.prototxt'
WEIGHT_v21_SMALL_PATH = 'midas_v2.1_small.onnx'
MODEL_v21_SMALL_PATH = 'midas_v2.1_small.onnx.prototxt'
REMOTE_PATH = 'https://storage.googleapis.com/ailia-models/midas/'

IMAGE_PATH = 'lights.jpeg'
SAVE_IMAGE_PATH = 'depth_lights_midas.png'
IMAGE_HEIGHT = 384
IMAGE_WIDTH = 384
IMAGE_HEIGHT_SMALL = 256
IMAGE_WIDTH_SMALL = 256
IMAGE_MULTIPLE_OF = 32

# ======================
# Argument Parser Config
# ======================
parser = get_base_parser('MiDaS model', IMAGE_PATH, SAVE_IMAGE_PATH)
parser.add_argument(
    '-v21', '--version21', dest='v21', action='store_true',
    help='Use model version 2.1.'
)
parser.add_argument(
    '-t', '--model_type', default='small', choices=('large', 'small'),
    help='model type: large or small. small can be specified only for version 2.1 model.'
)
args = update_parser(parser)

# ======================
# Helper functions for histogram and overlay
# ======================
def plot_histogram(depth_map):
    """Plot the histogram of depth values."""
    plt.figure(figsize=(10, 5))
    plt.hist(depth_map.flatten(), bins=256, range=(depth_map.min(), depth_map.max()), color='blue', alpha=0.7)
    plt.title('Histogram of Depth Values')
    plt.xlabel('Depth Value')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.show()

def overlay_depth_on_image(original_image, depth_map, alpha=0.6):
    """Overlay the depth map on the original image."""
    depth_normalized = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    depth_colormap = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_JET)
    
    depth_colormap_resized = cv2.resize(depth_colormap, (original_image.shape[1], original_image.shape[0]))
    overlay_img = cv2.addWeighted(original_image, alpha, depth_colormap_resized, 1 - alpha, 0)
    
    return overlay_img

# ======================
# Depth Uncertainty Functions
# ======================
def calculate_depth_uncertainty(depth_map, window_size=5):
    """Calculate the uncertainty (variance) of depth values using a sliding window."""
    # Create an empty array for uncertainty values
    uncertainty_map = np.zeros_like(depth_map)
    
    # Define the window for calculating local variance
    pad_size = window_size // 2
    padded_depth = np.pad(depth_map, pad_size, mode='reflect')
    
    # Calculate the local variance in the window for each pixel
    for i in range(pad_size, depth_map.shape[0] + pad_size):
        for j in range(pad_size, depth_map.shape[1] + pad_size):
            local_window = padded_depth[i - pad_size:i + pad_size + 1, j - pad_size:j + pad_size + 1]
            local_variance = np.var(local_window)
            uncertainty_map[i - pad_size, j - pad_size] = local_variance

    return uncertainty_map

def overlay_uncertainty_on_image(original_image, uncertainty_map, alpha=0.6):
    """Overlay the uncertainty map on the original image."""
    uncertainty_normalized = cv2.normalize(uncertainty_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    uncertainty_colormap = cv2.applyColorMap(uncertainty_normalized, cv2.COLORMAP_HOT)
    
    uncertainty_colormap_resized = cv2.resize(uncertainty_colormap, (original_image.shape[1], original_image.shape[0]))
    overlay_img = cv2.addWeighted(original_image, alpha, uncertainty_colormap_resized, 1 - alpha, 0)
    
    return overlay_img

# ======================
# Main functions
# ======================
def constrain_to_multiple_of(x, min_val=0, max_val=None):
    y = (np.round(x / IMAGE_MULTIPLE_OF) * IMAGE_MULTIPLE_OF).astype(int)
    if max_val is not None and y > max_val:
        y = (np.floor(x / IMAGE_MULTIPLE_OF) * IMAGE_MULTIPLE_OF).astype(int)
    if y < min_val:
        y = (np.ceil(x / IMAGE_MULTIPLE_OF) * IMAGE_MULTIPLE_OF).astype(int)
    return y

def midas_resize(image, target_height, target_width):
    h, w, _ = image.shape
    scale_height = target_height / h
    scale_width = target_width / w
    if scale_width < scale_height:
        scale_height = scale_width
    else:
        scale_width = scale_height
    new_height = constrain_to_multiple_of(
        scale_height * h, max_val=target_height
    )
    new_width = constrain_to_multiple_of(
        scale_width * w, max_val=target_width
    )
    return cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_CUBIC)

def midas_imread(image_path):
    if not os.path.isfile(image_path):
        logger.error(f'{image_path} not found.')
        sys.exit()
    image = imread(image_path)
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = normalize_image(image, 'ImageNet')

    h, w = (IMAGE_HEIGHT, IMAGE_WIDTH) if not args.v21 or args.model_type == 'large' \
               else (IMAGE_HEIGHT_SMALL, IMAGE_WIDTH_SMALL)
    return midas_resize(image, h, w)

def recognize_from_image(net):
    # Input image loop
    for image_path in args.input:
        logger.info(image_path)

        # Prepare input data
        img = midas_imread(image_path)
        img = img.transpose((2, 0, 1))  # Channel first
        img = img[np.newaxis, :, :, :]

        logger.debug(f'input image shape: {img.shape}')
        net.set_input_shape(img.shape)

        # Inference
        logger.info('Start inference...')
        result = net.predict(img)  # Predict depth map

        depth_min = result.min()
        depth_max = result.max()
        max_val = (2 ** 16) - 1

        if depth_max - depth_min > np.finfo("float").eps:
            out = max_val * (result - depth_min) / (depth_max - depth_min)
        else:
            out = 0

        depth_map = out[0]  # Extract depth map
        
        # Plot histogram of depth values
        plot_histogram(depth_map)

        # Overlay depth map on original image
        original_image = imread(image_path)
        overlay_img = overlay_depth_on_image(original_image, depth_map)
        
        # Display and save overlay
        cv2.imshow('Overlay Depth Map', overlay_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows() 
        cv2.imwrite('overlay_result.png', overlay_img)

        # Calculate and visualize depth uncertainty
        uncertainty_map = calculate_depth_uncertainty(depth_map, window_size=5)
        uncertainty_overlay = overlay_uncertainty_on_image(original_image, uncertainty_map)

        # Display the uncertainty map overlay
        cv2.imshow('Overlay Uncertainty Map', uncertainty_overlay)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        cv2.imwrite('uncertainty_result.png', uncertainty_overlay)

        savepath = get_savepath(args.savepath, image_path, ext='.png')
        logger.info(f'saved at : {savepath}')
        cv2.imwrite(savepath, out.transpose(1, 2, 0).astype("uint16"))
    logger.info('Script finished successfully.')

def main():
    weight_path = (WEIGHT_v21_PATH if args.model_type == 'large' else WEIGHT_v21_SMALL_PATH) \
        if args.v21 else WEIGHT_v20_PATH
    model_path = (MODEL_v21_PATH if args.model_type == 'large' else MODEL_v21_SMALL_PATH) \
        if args.v21 else MODEL_v20_PATH

    check_and_download_models(weight_path, model_path, REMOTE_PATH)

    net = ailia.Net(model_path, weight_path, env_id=args.env_id)

    recognize_from_image(net)

if __name__ == '__main__':
    main()
