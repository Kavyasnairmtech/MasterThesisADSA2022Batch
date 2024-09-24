import sys
import os
import time

import numpy as np
import cv2
from PIL import Image

import ailia

# Add the path to the detic folder (one level up from clip)
sys.path.append(os.path.abspath('../object_detection'))

print("0.  before detic september import")
# Now you can import detic_final_code
from detic_final_code import get_detections
print("3. passed detic imports - ie - back inside clip")

# Add the path to the depth estimation folder for MiDaS
sys.path.append(os.path.abspath('../depth_estimation_midas'))
from midas_final_code import main as midas_depth_estimation  # Import the MiDaS script as a function

# import original modules
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__))))
from arg_utils import get_base_parser, update_parser, get_savepath
from model_utils import check_and_download_models
from detector_utils import load_image
from classifier_utils import plot_results, print_results
from math_utils import softmax
import webcamera_utils
from logging import getLogger

from simple_tokenizer import SimpleTokenizer as _Tokenizer

print("5. passed clip imports ****************************************************")

logger = getLogger(__name__)

_tokenizer = _Tokenizer()

# ======================
# Parameters
# ======================

WEIGHT_VITB32_IMAGE_PATH = 'ViT-B32-encode_image.onnx'
MODEL_VITB32_IMAGE_PATH = 'ViT-B32-encode_image.onnx.prototxt'
WEIGHT_VITB32_TEXT_PATH = 'ViT-B32-encode_text.onnx'
MODEL_VITB32_TEXT_PATH = 'ViT-B32-encode_text.onnx.prototxt'
WEIGHT_VITL14_IMAGE_PATH = 'ViT-L14-encode_image.onnx'
MODEL_VITL14_IMAGE_PATH = 'ViT-L14-encode_image.onnx.prototxt'
WEIGHT_VITL14_TEXT_PATH = 'ViT-L14-encode_text.onnx'
MODEL_VITL14_TEXT_PATH = 'ViT-L14-encode_text.onnx.prototxt'
WEIGHT_RN50_IMAGE_PATH = 'RN50-encode_image.onnx'
MODEL_RN50_IMAGE_PATH = 'RN50-encode_image.onnx.prototxt'
WEIGHT_RN50_TEXT_PATH = 'RN50-encode_text.onnx'
MODEL_RN50_TEXT_PATH = 'RN50-encode_text.onnx.prototxt'
REMOTE_PATH = 'https://storage.googleapis.com/ailia-models/clip/'

global IMAGE_PATH, SAVE_IMAGE_PATH
IMAGE_PATH = 'lights.jpeg'
SAVE_IMAGE_PATH = 'clip_lights_sep24_output.png'

IMAGE_SIZE = 224

# ======================
# Argument Parser Config
# ======================

parser = get_base_parser(
    'CLIP', IMAGE_PATH, SAVE_IMAGE_PATH
)
parser.add_argument(
    '-t', '--text', dest='text_inputs', type=str,
    action='append',
    help='Input text. (can be specified multiple times)'
)
parser.add_argument(
    '--desc_file', default=None, metavar='DESC_FILE', type=str,
    help='description file'
)
parser.add_argument(
    '-m', '--model_type', default='RN50', choices=('ViTB32', 'ViTL14', 'RN50'),
    help='model type'
)
parser.add_argument(
    '--onnx',
    action='store_true',
    help='execute onnxruntime version.'
)
args = update_parser(parser)

# ======================
# Main functions
# ======================

def tokenize(texts, context_length=77, truncate=False):
    if isinstance(texts, str):
        texts = [texts]

    sot_token = _tokenizer.encoder["<|startoftext|>"]
    eot_token = _tokenizer.encoder["<|endoftext|>"]
    all_tokens = [[sot_token] + _tokenizer.encode(text) + [eot_token] for text in texts]
    result = np.zeros((len(all_tokens), context_length), dtype=np.int64)

    for i, tokens in enumerate(all_tokens):
        if len(tokens) > context_length:
            if truncate:
                tokens = tokens[:context_length]
                tokens[-1] = eot_token
            else:
                raise RuntimeError(f"Input {texts[i]} is too long for context length {context_length}")

        result[i, :len(tokens)] = np.array(tokens)

    result = result.astype(np.int64)

    return result


def preprocess(img):
    h, w = (IMAGE_SIZE, IMAGE_SIZE)
    im_h, im_w, _ = img.shape

    # resize
    scale = h / min(im_h, im_w)
    ow, oh = round(im_w * scale), round(im_h * scale)
    if ow != im_w or oh != im_h:
        img = np.array(Image.fromarray(img).resize((ow, oh), Image.BICUBIC))

    # center_crop
    if ow > w:
        x = (ow - w) // 2
        img = img[:, x:x + w, :]
    if oh > h:
        y = (oh - h) // 2
        img = img[y:y + h, :, :]

    img = img[:, :, ::-1]  # BGR -> RBG
    img = img / 255

    mean = np.array((0.48145466, 0.4578275, 0.40821073))
    std = np.array((0.26862954, 0.26130258, 0.27577711))
    img = (img - mean) / std

    img = img.transpose(2, 0, 1)  # HWC -> CHW
    img = np.expand_dims(img, axis=0)

    # print(f"********************************************* Preprocessed image shape: {img.shape} *********************************************")

    img = img.astype(np.float32)

    return img


def predict(net, img, text_feature):
    print(f"Image shape is : {img.shape}")

    if len(img.shape) == 3:  # Typically (H, W, C), like (960, 720, 3)
        print("Image has 3 dimensions -> Preprocessing image.")
        img = preprocess(img)  # Perform preprocessing
    elif len(img.shape) == 4:  # Typically (1, 3, H, W), like (1, 3, 224, 224)
        print("Image has 4 dimensions -> Already preprocessed, skipping.")
    else:
        print(f"Image shape is {img.shape} -> Unrecognized shape, skipping preprocessing.")
    # img = preprocess(img)

    # feedforward
    if not args.onnx:
        output = net.predict([img])
    else:
        output = net.run(None, {'image': img})

    image_feature = output[0]

    image_feature = image_feature / np.linalg.norm(image_feature, ord=2, axis=-1, keepdims=True)

    logit_scale = 100
    logits_per_image = (image_feature * logit_scale).dot(text_feature.T)

    pred = softmax(logits_per_image, axis=1)

    return pred[0]


def predict_text_feature(net, text):
    text_tokens = tokenize(text)

    # feedforward
    text_feature = []
    batch_size_limit = 16

    for i in range(0, text_tokens.shape[0], batch_size_limit):
        batch_size = min(batch_size_limit, text_tokens.shape[0] - i)
        logger.info("Embedding " + str(i) + " to " + str(i+batch_size))
        if not args.onnx:
            output = net.predict([text_tokens[i:i+batch_size,:]])
        else:
            output = net.run(None, {'text': text_tokens[i:i+batch_size,:]})
        text_feature.append(output[0])

    text_feature = np.concatenate(text_feature)

    text_feature = text_feature / np.linalg.norm(text_feature, ord=2, axis=-1, keepdims=True)

    return text_feature

def run_midas():
    """Run MiDaS depth estimation using predefined IMAGE_PATH and SAVE_IMAGE_PATH."""
    logger.info(f"Running MiDaS depth estimation ")

    # Set the input image path and output save path directly
    IMAGE_PATH = 'lights.jpeg'  # Set the input image
    SAVE_IMAGE_PATH_MIDAS = 'depth_lights_out_midas.png'  # Set the desired output file path

    # Call the MiDaS depth estimation function
    midas_depth_estimation()

    logger.info(f"MiDaS depth estimation completed and saved to {SAVE_IMAGE_PATH_MIDAS}.")


def recognize_from_image(net_image, net_text):
    text_inputs = args.text_inputs
    desc_file = args.desc_file
    if desc_file:
        with open(desc_file) as f:
            text_inputs = [x.strip() for x in f.readlines() if x.strip()]
    elif text_inputs is None:
        text_inputs = [f"a {c}" for c in ("a large blue plastic trash can with a lid", "dog", "cat")]

    text_feature = predict_text_feature(net_text, text_inputs)

    # input image loop
    for image_path in args.input:
        logger.info(image_path)

        # Run MiDaS depth estimation for the current image
        run_midas()  # This will trigger the MiDaS script

        # prepare input data
        img = load_image(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

        # inference
        logger.info('Start inference...')
        pred = predict(net_image, img, text_feature)

        # show results
        pred = np.expand_dims(pred, axis=0)
        print_results(pred, text_inputs)

    logger.info('Script finished successfully.')


def save_image_patch(img, bbox, filename="best_patch.jpg"):
    x0, y0, x1, y1 = map(int, bbox)
    patch = img[y0:y1, x0:x1]
    cv2.imwrite(filename, patch)
    logger.info(f"Best matched patch saved as {filename}")

def generate_visual_embedding(net_image, patch):
    """
    Generate visual embedding from a preprocessed image patch using the CLIP model.
    
    Args:
        net_image: The CLIP image model instance.
        patch: The preprocessed image patch with shape (1, 3, 224, 224).
        
    Returns:
        visual_embedding: A normalized visual embedding with shape (512,).
    """
    # Feed the preprocessed patch into the CLIP model
    if not args.onnx:
        # Using Ailia (or other framework), pass the patch through the model
        output = net_image.predict([patch])
    else:
        # Using ONNX runtime
        output = net_image.run(None, {'image': patch})
    
    # Extract the visual embedding from the model output
    visual_embedding = output[0]  # Usually the output is [batch_size, embedding_dim], we take the first batch

    # Normalize the embedding to unit length (L2 norm)
    visual_embedding = visual_embedding / np.linalg.norm(visual_embedding, ord=2, axis=-1, keepdims=True)
    
    # Ensure the visual embedding is 1D
    visual_embedding = visual_embedding.flatten()

    # print(f"Generated visual embedding shape: {visual_embedding.shape}")
    
    return visual_embedding

# def softmax(x):
#     """Compute softmax values for each set of scores in x."""
#     exp_x = np.exp(x - np.max(x))  # Subtract max for numerical stability
#     return exp_x / np.sum(exp_x, axis=0)

def save_patch_with_label(img, bbox, label, output_dir="patches", label_counter=None):
    """
    Save an image patch along with the Detic-identified label as the filename.
    """
    x0, y0, x1, y1 = map(int, bbox)
    patch = img[y0:y1, x0:x1]

    # Create a safe label by replacing spaces and limiting length
    safe_label = label.replace(" ", "_").replace("%", "")
    if len(safe_label) > 50:
        safe_label = safe_label[:50]

    # output_dir = "patches"
    output_dir = os.path.join(os.getcwd(), output_dir)
    os.makedirs(output_dir, exist_ok=True)

    # Initilaize label counter ifnot providede earlier
    if label_counter is None:
        label_counter = {}

    # Check if the label already exists in the counter
    if safe_label not in label_counter:
        label_counter[safe_label] = 1
    else:
        label_counter[safe_label] += 1
    
    # # Save the patch with the label as the file name
    # os.makedirs(output_dir, exist_ok=True)
    # filename = f"{output_dir}/{safe_label}.jpg"
    # cv2.imwrite(filename, patch)
    # logger.info(f"Patch saved as {filename}")

    # Append the counter to the label for the filename
    filename = f"{output_dir}/{safe_label}_{label_counter[safe_label]}.jpg"

    # Save the patch with the updated filename
    cv2.imwrite(filename, patch)
    logger.info(f"Patch saved as {filename}")

    return label_counter  # Return the updated label counter

def match_visual_patches_with_text(net_image, net_text, image_path, text, similarity_threshold=0.20):
    label_counter = {}

    # Step 1: Generate the text embedding
    text_embedding = predict_text_feature(net_text, text)
    logger.info("Text embedding generated for: " + text)
    
    # Step 2: Get detections from Detic
    pred_boxes, scores, pred_classes, labels = get_detections(image_path)
    # print(f"Pred boxes from detic: {pred_boxes}")
    logger.info(f"Detected {len(pred_boxes)} objects")
    
    # Step 3: Load image and get CLIP embeddings for visual patches
    img = load_image(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    
    # best_match_score = -1
    # best_match_bbox = None

    matched_patches = []  # To store patches that exceed the similarity threshold

    # Step 4: Save patches with Detic-identified labels
    for i, (box, label) in enumerate(zip(pred_boxes, labels)):
        label_counter = save_patch_with_label(img, box, label, label_counter=label_counter)
    
    # Step 4: For each detected box, generate visual embeddings
    for i, box in enumerate(pred_boxes):
        x0, y0, x1, y1 = map(int, box)
        patch = img[y0:y1, x0:x1]
        
        # Resize patch to the size required by CLIP model before preprocessing
        # print(f"Old Patch size: {patch.size}")
        # print(f"Old Patch shape: {patch.shape}")
        cv2.imwrite(f"patch_{i}.jpg", patch)

        # Resize the patch before preprocessing
        patch = cv2.resize(patch, (IMAGE_SIZE, IMAGE_SIZE))
        # print(f"Resized Patch size: {patch.size}")
        # print(f"Resized Patch shape: {patch.shape}")
        
        # Preprocess the resized patch
        patch = preprocess(patch)
        
        # print(f"New Patch size after preprocess: {patch.size}")
        # print(f"New Patch shape after preprocess: {patch.shape}")

        # Step 7: Generate visual embedding for the patch
        visual_embedding = generate_visual_embedding(net_image, patch)
        # print(f"Visual embedding: {visual_embedding}, shape: {visual_embedding.shape}")
        
        # Ensure embeddings are 1D
        visual_embedding = visual_embedding.flatten()
        text_embedding = text_embedding.flatten()

        # similarity measurements
        # dot product
        similarity = np.dot(visual_embedding, text_embedding)

    # Track the patches that meet or exceed the similarity threshold
        if similarity > similarity_threshold:
            matched_patches.append((box, similarity))
            logger.info(f"Similarity for patch {i}: {similarity} -> Exceeds threshold")

    # Step 5: Save the matched patches
    if len(matched_patches) > 0:
        for i, (best_match_bbox, similarity) in enumerate(matched_patches):
            filename = f"best_patch_{i+1}.jpg"  # Save each matched patch with a unique filename
            save_image_patch(img, best_match_bbox, filename=filename)
            logger.info(f"Matched patch saved as {filename} with similarity {similarity}")
    else:
        logger.info("No matches found that exceed the similarity threshold")
        print("No matches found that exceed the similarity threshold")


def recognize_from_video(net_image, net_text):
    text_inputs = args.text_inputs
    desc_file = args.desc_file
    if desc_file:
        with open(desc_file) as f:
            text_inputs = [x.strip() for x in f.readlines() if x.strip()]
    elif text_inputs is None:
        text_inputs = [f"a {c}" for c in ("human", "dog", "cat")]

    text_feature = predict_text_feature(net_text, text_inputs)

    capture = webcamera_utils.get_capture(args.video)
    # create video writer if savepath is specified as video format
    if args.savepath is not None:
        f_h = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        f_w = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        writer = webcamera_utils.get_writer(args.savepath, f_h, f_w)
    else:
        writer = None

    frame_shown = False
    while (True):
        ret, frame = capture.read()
        if (cv2.waitKey(1) & 0xFF == ord('q')) or not ret:
            break
        if frame_shown and cv2.getWindowProperty('frame', cv2.WND_PROP_VISIBLE) == 0:
            break

        img = frame

        pred = predict(net_image, img, text_feature)

        plot_results(frame, np.expand_dims(pred, axis=0), text_inputs)

        cv2.imshow('frame', frame)
        frame_shown = True

        # save results
        if writer is not None:
            writer.write(frame)

    capture.release()
    cv2.destroyAllWindows()
    if writer is not None:
        writer.release()

    logger.info('Script finished successfully.')

def main():
    dic_model = {
        'ViTB32': (
            (WEIGHT_VITB32_IMAGE_PATH, MODEL_VITB32_IMAGE_PATH),
            (WEIGHT_VITB32_TEXT_PATH, MODEL_VITB32_TEXT_PATH)),
        'ViTL14': (
            (WEIGHT_VITL14_IMAGE_PATH, MODEL_VITL14_IMAGE_PATH),
            (WEIGHT_VITL14_TEXT_PATH, MODEL_VITL14_TEXT_PATH)),
        'RN50': (
            (WEIGHT_RN50_IMAGE_PATH, MODEL_RN50_IMAGE_PATH),
            (WEIGHT_RN50_TEXT_PATH, MODEL_RN50_TEXT_PATH)),
    }
    (WEIGHT_IMAGE_PATH, MODEL_IMAGE_PATH), (WEIGHT_TEXT_PATH, MODEL_TEXT_PATH) = dic_model[args.model_type]

    # model files check and download
    logger.info('Checking encode_image model...')
    check_and_download_models(WEIGHT_IMAGE_PATH, MODEL_IMAGE_PATH, REMOTE_PATH)
    logger.info('Checking encode_text model...')
    check_and_download_models(WEIGHT_TEXT_PATH, MODEL_TEXT_PATH, REMOTE_PATH)

    env_id = args.env_id

    # disable FP16
    if "FP16" in ailia.get_environment(args.env_id).props or sys.platform == 'Darwin':
        logger.warning('This model do not work on FP16. So use CPU mode.')
        args.env_id = 0

    # initialize
    if not args.onnx:
        memory_mode = ailia.get_memory_mode(
            reduce_constant=True, ignore_input_with_initializer=True,
            reduce_interstage=False, reuse_interstage=False)
        net_image = ailia.Net(MODEL_IMAGE_PATH, WEIGHT_IMAGE_PATH, env_id=env_id, memory_mode=memory_mode)
        net_text = ailia.Net(MODEL_TEXT_PATH, WEIGHT_TEXT_PATH, env_id=env_id, memory_mode=memory_mode)
    else:
        import onnxruntime
        net_image = onnxruntime.InferenceSession(WEIGHT_IMAGE_PATH)
        net_text = onnxruntime.InferenceSession(WEIGHT_TEXT_PATH)

    if args.video is not None:
        # video mode
        recognize_from_video(net_image, net_text)
    else:
        # image mode
        recognize_from_image(net_image, net_text)

    text_input = 'a socket'
    match_visual_patches_with_text(net_image, net_text, IMAGE_PATH, text_input, similarity_threshold=0.20)

if __name__ == '__main__':
    main()
