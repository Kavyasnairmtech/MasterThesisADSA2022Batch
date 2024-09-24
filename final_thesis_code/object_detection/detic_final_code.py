import sys
import time
import os
import datetime
import platform

import numpy as np
import cv2
from PIL import Image

import ailia

# import original modules
# sys.path.append('../../util')
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

print("1. inside detic september ")
from arg_utils import get_base_parser, update_parser, get_savepath  # noqa
from model_utils import check_and_download_models  # noqa
from detector_utils import load_image  # noqa
from webcamera_utils import get_capture, get_writer  # noqa
from functional import grid_sample  # noqa
# logger
from logging import getLogger  # noqa

from dataset_utils import get_lvis_meta_v1, get_in21k_meta_v1
from color_utils import random_color, color_brightness

print("2. passed detic september's all imports ************************************************* ")

logger = getLogger(__name__)

# ======================
# Parameters
# ======================

WEIGHT_SWINB_LVIS_PATH = 'Detic_C2_SwinB_896_4x_IN-21K+COCO_lvis.onnx'
MODEL_SWINB_LVIS_PATH = 'Detic_C2_SwinB_896_4x_IN-21K+COCO_lvis.onnx.prototxt'
WEIGHT_SWINB_IN21K_PATH = 'Detic_C2_SwinB_896_4x_IN-21K+COCO_in21k.onnx'
MODEL_SWINB_IN21K_PATH = 'Detic_C2_SwinB_896_4x_IN-21K+COCO_in21k.onnx.prototxt'
WEIGHT_R50_LVIS_PATH = 'Detic_C2_R50_640_4x_lvis.onnx'
MODEL_R50_LVIS_PATH = 'Detic_C2_R50_640_4x_lvis.onnx.prototxt'
WEIGHT_R50_IN21K_PATH = 'Detic_C2_R50_640_4x_in21k.onnx'
MODEL_R50_IN21K_PATH = 'Detic_C2_R50_640_4x_in21k.onnx.prototxt'
REMOTE_PATH = 'https://storage.googleapis.com/ailia-models/detic/'

IMAGE_PATH = 'lights.jpeg'
SAVE_IMAGE_PATH = 'output_lights_sep_24_morning.png'

# ======================
# Arguemnt Parser Config
# ======================

parser = get_base_parser(
    'Detic', IMAGE_PATH, SAVE_IMAGE_PATH
)
parser.add_argument(
    '--seed', type=int, default=int(datetime.datetime.now().strftime('%Y%m%d')),
    help='random seed for selection the color of the box'
)
parser.add_argument(
    '-m', '--model_type', default='SwinB_896_4x', choices=('SwinB_896_4x', 'R50_640_4x'),
    help='model type'
)
parser.add_argument(
    '-vc', '--vocabulary', default='in21k', choices=('lvis', 'in21k'),
    help='vocabulary'
)
parser.add_argument(
    '--opset16',
    action='store_true',
    help='Use the opset16 model. In that case, grid_sampler runs inside the model.'
)
parser.add_argument(
    '--onnx',
    action='store_true',
    help='execute onnxruntime version.'
)
parser.add_argument(
    '-dw', '--detection_width',
    default=800, type=int,   # tempolary limit to 800px (original : 1333)
    help='The detection width for detic. (default: 800)'
)
args = update_parser(parser)


# ======================
# Prediction Function
# ======================

def do_paste_mask(masks, boxes, im_h, im_w):
    """
    Args:
        masks: N, 1, H, W
        boxes: N, 4
        img_h, img_w (int):
        skip_empty (bool): only paste masks within the region that
            tightly bound all boxes, and returns the results this region only.
            An important optimization for CPU.

    Returns:
        if skip_empty == False, a mask of shape (N, img_h, img_w)
        if skip_empty == True, a mask of shape (N, h', w'), and the slice
            object for the corresponding region.
    """

    x0_int, y0_int = 0, 0
    x1_int, y1_int = im_w, im_h
    x0, y0, x1, y1 = np.split(boxes, 4, axis=1)  # each is Nx1

    img_y = np.arange(y0_int, y1_int, dtype=np.float32) + 0.5
    img_x = np.arange(x0_int, x1_int, dtype=np.float32) + 0.5
    img_y = (img_y - y0) / (y1 - y0) * 2 - 1
    img_x = (img_x - x0) / (x1 - x0) * 2 - 1

    gx = np.repeat(img_x[:, None, :], img_y.shape[1], axis=1)
    gy = np.repeat(img_y[:, :, None], img_x.shape[1], axis=2)
    grid = np.stack([gx, gy], axis=3)

    img_masks = grid_sample(masks, grid, align_corners=False)

    return img_masks[:, 0]


def paste_masks_in_image(
        masks, boxes, image_shape, threshold: float = 0.5):
    """
    Paste a set of masks that are of a fixed resolution (e.g., 28 x 28) into an image.
    The location, height, and width for pasting each mask is determined by their
    corresponding bounding boxes in boxes.

    Note:
        This is a complicated but more accurate implementation. In actual deployment, it is
        often enough to use a faster but less accurate implementation.
        See :func:`paste_mask_in_image_old` in this file for an alternative implementation.
    """

    if len(masks) == 0:
        return np.zeros((0,) + image_shape, dtype=np.uint8)

    im_h, im_w = image_shape

    img_masks = do_paste_mask(
        masks[:, None, :, :], boxes, im_h, im_w,
    )
    img_masks = img_masks >= threshold

    return img_masks


def mask_to_polygons(mask):
    # cv2.RETR_CCOMP flag retrieves all the contours and arranges them to a 2-level
    # hierarchy. External contours (boundary) of the object are placed in hierarchy-1.
    # Internal contours (holes) are placed in hierarchy-2.
    # cv2.CHAIN_APPROX_NONE flag gets vertices of polygons from contours.

    mask = np.ascontiguousarray(mask)  # some versions of cv2 does not support incontiguous arr
    res = cv2.findContours(mask.astype("uint8"), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    hierarchy = res[-1]
    if hierarchy is None:  # empty mask
        return [], False

    has_holes = (hierarchy.reshape(-1, 4)[:, 3] >= 0).sum() > 0
    res = res[-2]
    res = [x.flatten() for x in res]

    # These coordinates from OpenCV are integers in range [0, W-1 or H-1].
    # We add 0.5 to turn them into real-value coordinate space. A better solution
    # would be to first +0.5 and then dilate the returned polygon by 0.5.
    res = [x + 0.5 for x in res if len(x) >= 6]

    return res, has_holes


def draw_predictions(img, predictions):
    vocabulary = args.vocabulary

    height, width = img.shape[:2]

    boxes = predictions["pred_boxes"].astype(np.int64)
    scores = predictions["scores"]
    classes = predictions["pred_classes"].tolist()
    masks = predictions["pred_masks"].astype(np.uint8)

    class_names = (
        get_lvis_meta_v1() if vocabulary == 'lvis' else get_in21k_meta_v1()
    )["thing_classes"]
    # labels = [class_names[i] for i in classes] # onnx runtime
    labels = [class_names[int(i)] for i in classes]  # ailia always returns float tensor so need to add cast
    labels_with_scores = ["{} {:.0f}%".format(l, s * 100) for l, s in zip(labels, scores)]

    num_instances = len(boxes)

    np.random.seed(args.seed)
    assigned_colors = [random_color(maximum=255) for _ in range(num_instances)]

    areas = np.prod(boxes[:, 2:] - boxes[:, :2], axis=1)
    if areas is not None:
        sorted_idxs = np.argsort(-areas).tolist()
        # Re-order overlapped instances in descending order.
        boxes = boxes[sorted_idxs]
        labels_with_scores = [labels_with_scores[k] for k in sorted_idxs]
        masks = [masks[idx] for idx in sorted_idxs]
        assigned_colors = [assigned_colors[idx] for idx in sorted_idxs]

    default_font_size = int(max(np.sqrt(height * width) // 90, 10))

    for i in range(num_instances):
        color = assigned_colors[i]
        color = (int(color[0]), int(color[1]), int(color[2]))
        img_b = img.copy()

        # draw box
        x0, y0, x1, y1 = boxes[i]
        cv2.rectangle(
            img_b, (x0, y0), (x1, y1),
            color=color,
            thickness=default_font_size // 4)

        # draw segment
        polygons, _ = mask_to_polygons(masks[i])
        for points in polygons:
            points = np.array(points).reshape((1, -1, 2)).astype(np.int32)
            cv2.fillPoly(img_b, pts=[points], color=color)

        img = cv2.addWeighted(img, 0.5, img_b, 0.5, 0)

    for i in range(num_instances):
        color = assigned_colors[i]
        color_text = color_brightness(color, brightness_factor=0.7)

        color = (int(color[0]), int(color[1]), int(color[2]))
        color_text = (int(color_text[0]), int(color_text[1]), int(color_text[2]))

        x0, y0, x1, y1 = boxes[i]

        SMALL_OBJECT_AREA_THRESH = 1000
        instance_area = (y1 - y0) * (x1 - x0)

        # for small objects, draw text at the side to avoid occlusion
        text_pos = (x0, y0)  # if drawing boxes, put text on the box corner.
        if instance_area < SMALL_OBJECT_AREA_THRESH or y1 - y0 < 40:
            if y1 >= height - 5:
                text_pos = (x1, y0)
            else:
                text_pos = (x0, y1)

        # draw label
        x, y = text_pos
        text = labels_with_scores[i]
        font = cv2.FONT_HERSHEY_SIMPLEX
        height_ratio = (y1 - y0) / np.sqrt(height * width)
        font_scale = (
                np.clip((height_ratio - 0.02) / 0.08 + 1, 1.2, 2) * 0.5)
        font_thickness = 1
        text_size, _ = cv2.getTextSize(text, font, font_scale, font_thickness)
        text_w, text_h = text_size
        cv2.rectangle(img, text_pos, (int(x + text_w * 0.6), y + text_h), (0, 0, 0), -1)
        cv2.putText(
            img, text, (x, y + text_h - 5),
            fontFace=font,
            fontScale=font_scale * 0.6,
            color=color_text,
            thickness=font_thickness,
            lineType=cv2.LINE_AA)

    return img, labels

def save_image_with_labels(img, labels, output_dir, image_prefix="output"):
    # Create a safe label by joining them with underscores and limiting length
    safe_labels = "_".join(labels).replace(" ", "_").replace("%", "")
    if len(safe_labels) > 100:  # Limit the file name length
        safe_labels = safe_labels[:100]

    # Create the file name using the labels
    filename = f"{image_prefix}_{safe_labels}.jpg"
    save_path = os.path.join(output_dir, filename)

    # Save the image
    cv2.imwrite(save_path, img)
    print(f"Image saved with filename: {filename}")


def preprocess(img):
    im_h, im_w, _ = img.shape

    img = img[:, :, ::-1]  # BGR -> RGB
    print("inside preprocess")

    size = args.detection_width
    max_size = args.detection_width
    scale = size / min(im_h, im_w)
    if im_h < im_w:
        oh, ow = size, scale * im_w
    else:
        oh, ow = scale * im_h, size
    if max(oh, ow) > max_size:
        scale = max_size / max(oh, ow)
        oh = oh * scale
        ow = ow * scale
    ow = int(ow + 0.5)
    oh = int(oh + 0.5)

    img = np.asarray(Image.fromarray(img).resize((ow, oh), Image.BILINEAR))

    img = img.transpose((2, 0, 1))  # HWC -> CHW
    img = np.expand_dims(img, axis=0)
    img = img.astype(np.float32)

    return img

def post_processing(
        pred_boxes, scores, pred_classes, pred_masks, im_hw, pred_hw):
    scale_x, scale_y = (
        im_hw[1] / pred_hw[1],
        im_hw[0] / pred_hw[0],
    )

    pred_boxes[:, 0::2] *= scale_x
    pred_boxes[:, 1::2] *= scale_y
    pred_boxes[:, [0, 2]] = np.clip(pred_boxes[:, [0, 2]], 0, im_hw[1])
    pred_boxes[:, [1, 3]] = np.clip(pred_boxes[:, [1, 3]], 0, im_hw[0])

    threshold = 0
    widths = pred_boxes[:, 2] - pred_boxes[:, 0]
    heights = pred_boxes[:, 3] - pred_boxes[:, 1]
    keep = (widths > threshold) & (heights > threshold)

    pred_boxes = pred_boxes[keep]
    scores = scores[keep]
    pred_classes = pred_classes[keep]
    pred_masks = pred_masks[keep]

    mask_threshold = 0.5
    pred_masks = paste_masks_in_image(
        pred_masks[:, 0, :, :], pred_boxes,
        (im_hw[0], im_hw[1]), mask_threshold
    )

    pred = {
        'pred_boxes': pred_boxes,
        'scores': scores,
        'pred_classes': pred_classes,
        'pred_masks': pred_masks,
    }
    return pred

def predict(net, img):
    im_h, im_w = img.shape[:2]
    img = preprocess(img)
    pred_hw = img.shape[-2:]
    im_hw = np.array([im_h, im_w]).astype(np.int64)

    # feedforward
    if args.opset16:
        if not args.onnx:
            output = net.predict([img, im_hw])
        else:
            output = net.run(None, {'img': img, 'im_hw': im_hw})
    else:
        if not args.onnx:
            output = net.predict([img])
        else:
            output = net.run(None, {'img': img})

    pred_boxes, scores, pred_classes, pred_masks = output

    if not args.opset16:
        pred = post_processing(
            pred_boxes, scores, pred_classes, pred_masks,
            (im_h, im_w), pred_hw
        )
    else:
        pred = {
            'pred_boxes': pred_boxes,
            'scores': scores,
            'pred_classes': pred_classes,
            'pred_masks': pred_masks,
        }

    return pred


# ======================
# Get Detections (for external call)
# ======================

def get_detections(image_path):
    img = load_image(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    
    # Initialize model
    net = load_model()

    # Perform prediction
    pred = predict(net, img)

    # Get bounding boxes, scores, and classes
    pred_boxes = pred['pred_boxes']
    scores = pred['scores']
    pred_classes = pred['pred_classes']

    # Map class IDs to human-readable class names
    if args.vocabulary == 'lvis':
        class_names = get_lvis_meta_v1()["thing_classes"]
    else:
        class_names = get_in21k_meta_v1()["thing_classes"]

    pred_labels = [class_names[int(cls_id)] for cls_id in pred_classes]

    return pred_boxes, scores, pred_classes, pred_labels


# ======================
# Recognize from Image (for direct call)
# ======================

def recognize_from_image(net):
    for image_path in args.input:
        logger.info(image_path)

        # prepare input data
        img = load_image(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

        # inference
        logger.info('Start inference...')
        pred = predict(net, img)

        # Print bounding boxes, scores, and object classes
        print("Bounding boxes:", pred['pred_boxes'])
        print("Confidence scores:", pred['scores'])
        print("Object classes:", pred['pred_classes'])

        # draw prediction
        res_img, detected_labels = draw_predictions(img, pred)

        # plot result
        savepath = get_savepath(args.savepath, image_path, ext='.png')
        logger.info(f'saved at : {savepath}')
        cv2.imwrite(savepath, res_img)

    logger.info('Script finished successfully.')


# ======================
# Model Loading Function
# ======================

def load_model():
    if args.opset16:
        dic_model = {
            ('SwinB_896_4x', 'lvis'): (WEIGHT_SWINB_LVIS_OP16_PATH, MODEL_SWINB_LVIS_OP16_PATH),
            ('SwinB_896_4x', 'in21k'): (WEIGHT_SWINB_IN21K_OP16_PATH, MODEL_SWINB_IN21K_OP16_PATH),
            ('R50_640_4x', 'lvis'): (WEIGHT_R50_LVIS_OP16_PATH, MODEL_R50_LVIS_OP16_PATH),
            ('R50_640_4x', 'in21k'): (WEIGHT_R50_IN21K_OP16_PATH, MODEL_R50_IN21K_OP16_PATH),
        }
    else:
        dic_model = {
            ('SwinB_896_4x', 'lvis'): (WEIGHT_SWINB_LVIS_PATH, MODEL_SWINB_LVIS_PATH),
            ('SwinB_896_4x', 'in21k'): (WEIGHT_SWINB_IN21K_PATH, MODEL_SWINB_IN21K_PATH),
            ('R50_640_4x', 'lvis'): (WEIGHT_R50_LVIS_PATH, MODEL_R50_LVIS_PATH),
            ('R50_640_4x', 'in21k'): (WEIGHT_R50_IN21K_PATH, MODEL_R50_IN21K_PATH),
        }
    key = (args.model_type, args.vocabulary)
    WEIGHT_PATH, MODEL_PATH = dic_model[key]

    # model files check and download
    check_and_download_models(WEIGHT_PATH, MODEL_PATH, REMOTE_PATH)

    # disable FP16
    if "FP16" in ailia.get_environment(args.env_id).props or platform.system() == 'Darwin':
        logger.warning('This model does not work on FP16. So use CPU mode.')
        args.env_id = 0

    # initialize model
    if not args.onnx:
        if args.env_id == 0:
            memory_mode = ailia.get_memory_mode(
                reduce_constant=True, ignore_input_with_initializer=True,
                reduce_interstage=False, reuse_interstage=True)
        else:
            memory_mode = ailia.get_memory_mode(
                reduce_constant=True, ignore_input_with_initializer=True,
                reduce_interstage=True, reuse_interstage=False)
        net = ailia.Net(MODEL_PATH, WEIGHT_PATH, env_id=args.env_id, memory_mode=memory_mode)
    else:
        import onnxruntime
        net = onnxruntime.InferenceSession(WEIGHT_PATH)

    return net


# ======================
# Main Function
# ======================

def main():

    print("Loading detic model")
    net = load_model()

    print("inside detic september main ************************************************* ")

    if args.video is not None:
        recognize_from_video(net)
    else:
        recognize_from_image(net)

    print("detic main completed ************************************************* ")


if __name__ == '__main__':
    main()
