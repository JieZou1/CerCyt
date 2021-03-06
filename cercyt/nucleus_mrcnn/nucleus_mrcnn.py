"""
Usage:

    # Train a new model starting from ImageNet weights
    python3 nucleus_mrcnn.py train --dataset=/path/to/dataset --subset=train --weights=imagenet

    # Train a new model starting from specific weights file
    python3 nucleus_mrcnn.py train --dataset=/path/to/dataset --subset=train --weights=/path/to/weights.h5

    # Resume training a model that you had trained earlier
    python3 nucleus_mrcnn.py train --dataset=/path/to/dataset --subset=train --weights=last

    # Generate submission file
    python3 nucleus_mrcnn.py detect --dataset=/path/to/dataset --subset=train --weights=<last or /path/to/weights.h5>

"""

# Set matplotlib backend
# This has to be done before other importa that might
# set it, but only if we're running in script mode
# rather than being imported.
import io
import json

if __name__ == '__main__':
    import matplotlib
    # Agg backend runs without a display
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

import os
import sys
import numpy as np
import datetime
import skimage.io
from pycocotools.coco import COCO
import pycocotools.mask

from imgaug import augmenters as iaa

from mrcnn import model as modellib
from mrcnn.config import Config
from mrcnn import utils
from mrcnn import visualize

# Import CerCyt
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))  # To find local version of the library

from cercyt.nucleus_mrcnn.kaggle_nucleus_dataset import VAL_IMAGE_IDS as VAL_IMAGE_IDS
from cercyt.nucleus_mrcnn.kaggle_nucleus_dataset import KaggleNucleusDataset
from cercyt.nucleus_mrcnn.nlm_nucleus_dataset import NlmNucleusDataset
from cercyt.config import DIR_NUCLEUS_MRCNN, DIR_DATA_SCIENCE_BOWL_2018, DIR_NLM_PATCH

# Root directory of the project
# ROOT_DIR = os.path.abspath(".")
ROOT_DIR = DIR_NUCLEUS_MRCNN

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")

DEFAULT_MODEL_DIR = os.path.join(ROOT_DIR, 'models')

DEFAULT_IMAGE_DIR = os.path.join(DIR_NLM_PATCH, '12XS00147')

# Results directory
# Save submission files here
RESULTS_DIR = os.path.join(ROOT_DIR, "results")

# Path to trained weights file
COCO_WEIGHTS_PATH = os.path.join(DEFAULT_MODEL_DIR, "mask_rcnn_coco.h5")
KAGGLE_ALL_WEIGHTS_PATH = os.path.join(DEFAULT_MODEL_DIR, "mask_rcnn_nucleus_0040-all_kaggle.h5")
KAGGLE_16_WEIGHTS_PATH = os.path.join(DEFAULT_MODEL_DIR, "mask_rcnn_nucleus_0040-16_kaggle.h5")



############################################################
#  Configurations
############################################################


class NucleusConfig(Config):
    """Configuration for training on the nucleus segmentation dataset."""
    # Give the configuration a recognizable name
    NAME = "nucleus"

    # Adjust depending on your GPU memory
    IMAGES_PER_GPU = 4

    # Number of classes (including background)
    NUM_CLASSES = 1 + 1  # Background + nucleus

    # Number of training and validation steps per epoch
    STEPS_PER_EPOCH = (657 - len(VAL_IMAGE_IDS)) // IMAGES_PER_GPU
    VALIDATION_STEPS = max(1, len(VAL_IMAGE_IDS) // IMAGES_PER_GPU)

    # Don't exclude based on confidence. Since we have two classes
    # then 0.5 is the minimum anyway as it picks between nucleus and BG
    DETECTION_MIN_CONFIDENCE = 0

    # Backbone network architecture
    # Supported values are: resnet50, resnet101
    BACKBONE = "resnet50"

    # Input image resizing
    # Random crops of size 512x512
    IMAGE_RESIZE_MODE = "crop"
    IMAGE_MIN_DIM = 512
    IMAGE_MAX_DIM = 512
    IMAGE_MIN_SCALE = 2.0

    # Length of square anchor side in pixels
    RPN_ANCHOR_SCALES = (8, 16, 32, 64, 128)

    # ROIs kept after non-maximum suppression (training and inference)
    POST_NMS_ROIS_TRAINING = 1000
    POST_NMS_ROIS_INFERENCE = 2000

    # Non-max suppression threshold to filter RPN proposals.
    # You can increase this during training to generate more propsals.
    RPN_NMS_THRESHOLD = 0.9

    # How many anchors per image to use for RPN training
    RPN_TRAIN_ANCHORS_PER_IMAGE = 64

    # Image mean (RGB)
    MEAN_PIXEL = np.array([43.53, 39.56, 48.22])

    # If enabled, resizes instance masks to a smaller size to reduce
    # memory load. Recommended when using high-resolution images.
    USE_MINI_MASK = True
    MINI_MASK_SHAPE = (56, 56)  # (height, width) of the mini-mask

    # Number of ROIs per image to feed to classifier/mask heads
    # The Mask RCNN paper uses 512 but often the RPN doesn't generate
    # enough positive proposals to fill this and keep a positive:negative
    # ratio of 1:3. You can increase the number of proposals by adjusting
    # the RPN NMS threshold.
    TRAIN_ROIS_PER_IMAGE = 128

    # Maximum number of ground truth instances to use in one image
    MAX_GT_INSTANCES = 200

    # Max number of final detections per image
    DETECTION_MAX_INSTANCES = 400


class NucleusInferenceConfig(NucleusConfig):
    # Set batch size to 1 to run one image at a time
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    # Don't resize imager for inferencing
    IMAGE_RESIZE_MODE = "pad64"
    # Non-max suppression threshold to filter RPN proposals.
    # You can increase this during training to generate more propsals.
    RPN_NMS_THRESHOLD = 0.7

############################################################
#  Training
############################################################


def train(model, dataset_dir, subset):
    """Train the model."""
    # Training dataset.
    dataset_train = KaggleNucleusDataset()
    dataset_train.load_nucleus(dataset_dir, subset)
    dataset_train.prepare()

    # Validation dataset
    dataset_val = KaggleNucleusDataset()
    # dataset_val.load_nucleus(dataset_dir, "val")
    dataset_val.load_nucleus(dataset_dir, "stage1_train_cervical_val")
    dataset_val.prepare()

    # Image augmentation
    # http://imgaug.readthedocs.io/en/latest/source/augmenters.html
    augmentation = iaa.SomeOf((0, 2), [
        iaa.Fliplr(0.5),
        iaa.Flipud(0.5),
        iaa.OneOf([iaa.Affine(rotate=90),
                   iaa.Affine(rotate=180),
                   iaa.Affine(rotate=270)]),
        iaa.Multiply((0.8, 1.5)),
        iaa.GaussianBlur(sigma=(0.0, 5.0))
    ])

    # *** This training schedule is an example. Update to your needs ***

    # If starting from imagenet, train heads only for a bit
    # since they have random weights
    # print("Train network heads")
    # model.train(dataset_train, dataset_val,
    #             learning_rate=config.LEARNING_RATE,
    #             epochs=80,
    #             augmentation=augmentation,
    #             layers='heads')

    print("Train all layers")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=100,
                augmentation=augmentation,
                layers='all')

############################################################
#  RLE Encoding
############################################################


def rle_encode(mask):
    """Encodes a mask in Run Length Encoding (RLE).
    Returns a string of space-separated values.
    """
    assert mask.ndim == 2, "Mask must be of shape [Height, Width]"
    # Flatten it column wise
    m = mask.T.flatten()
    # Compute gradient. Equals 1 or -1 at transition points
    g = np.diff(np.concatenate([[0], m, [0]]), n=1)
    # 1-based indicies of transition points (where gradient != 0)
    rle = np.where(g != 0)[0].reshape([-1, 2]) + 1
    # Convert second index in each pair to lenth
    rle[:, 1] = rle[:, 1] - rle[:, 0]
    return " ".join(map(str, rle.flatten()))


def rle_decode(rle, shape):
    """Decodes an RLE encoded list of space separated
    numbers and returns a binary mask."""
    rle = list(map(int, rle.split()))
    rle = np.array(rle, dtype=np.int32).reshape([-1, 2])
    rle[:, 1] += rle[:, 0]
    rle -= 1
    mask = np.zeros([shape[0] * shape[1]], np.bool)
    for s, e in rle:
        assert 0 <= s < mask.shape[0]
        assert 1 <= e <= mask.shape[0], "shape: {}  s {}  e {}".format(shape, s, e)
        mask[s:e] = 1
    # Reshape and transpose
    mask = mask.reshape([shape[1], shape[0]]).T
    return mask


def mask_to_rle(image_id, mask, scores):
    "Encodes instance masks to submission format."
    assert mask.ndim == 3, "Mask must be [H, W, count]"
    # If mask is empty, return line with image ID only
    if mask.shape[-1] == 0:
        return "{},".format(image_id)
    # Remove mask overlaps
    # Multiply each instance mask by its score order
    # then take the maximum across the last dimension
    order = np.argsort(scores)[::-1] + 1  # 1-based descending
    mask = np.max(mask * np.reshape(order, [1, 1, -1]), -1)
    # Loop over instance masks
    lines = []
    for o in order:
        m = np.where(mask == o, 1, 0)
        # Skip if empty
        if m.sum() == 0.0:
            continue
        rle = rle_encode(m)
        lines.append("{}, {}".format(image_id, rle))
    return "\n".join(lines)


############################################################
#  Detection
############################################################

def detect(model, dataset_dir, subset):
    """Run detection on images in the given directory."""
    print("Running on {}".format(dataset_dir))

    # Create directory
    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)
    submit_dir = "submit_{:%Y%m%dT%H%M%S}".format(datetime.datetime.now())
    submit_dir = os.path.join(RESULTS_DIR, submit_dir)
    os.makedirs(submit_dir)

    # Read dataset
    dataset = KaggleNucleusDataset()
    dataset.load_nucleus(dataset_dir, subset)
    dataset.prepare()
    # Load over images
    submission = []
    for image_id in dataset.image_ids:
        # Load image and run detection
        image = dataset.load_image(image_id)
        # Detect objects
        r = model.detect([image], verbose=0)[0]
        # Encode image to RLE. Returns a string of multiple lines
        source_id = dataset.image_info[image_id]["id"]
        rle = mask_to_rle(source_id, r["masks"], r["scores"])
        submission.append(rle)
        # Save image with masks
        visualize.display_instances(
            image, r['rois'], r['masks'], r['class_ids'],
            dataset.class_names, r['scores'],
            show_bbox=False, show_mask=False,
            title="Predictions")
        plt.savefig("{}/{}.png".format(submit_dir, dataset.image_info[image_id]["id"]))

    # Save to csv file
    submission = "ImageId,EncodedPixels\n" + "\n".join(submission)
    file_path = os.path.join(submit_dir, "submit.csv")
    with open(file_path, "w") as f:
        f.write(submission)
    print("Saved to ", submit_dir)


def segment(model, image_folder):

    # Read dataset
    dataset = NlmNucleusDataset()
    dataset.load_nucleus_from_patches(image_folder)
    dataset.prepare()

    json_path = 'results.json'
    with io.open(json_path, 'w', encoding='utf8') as output:
        print('Writing results to: %s' % json_path)
        output.write('[\n')

        # Load over images
        for i, image_id in enumerate(dataset.image_ids):
            """Run segmentation on an image."""

            image_path = dataset.image_info[image_id]['path']
            vis_path = image_path.replace('patches', 'MCRNN').replace('.bmp', '.jpg')
            rle_path = image_path.replace('patches', 'MCRNN').replace('.bmp', '.txt')

            # if os.path.exists(vis_path) and os.path.exists(rle_path):
            #     continue

            # Load image and run detection
            image = dataset.load_image(image_id)
            r = model.detect([image])[0]

            rois = r['rois']
            masks = r['masks']
            class_ids = r['class_ids']
            scores = r['scores']

            # save segmentation visualization
            vis_path = vis_path.replace('.tif', '.jpg')
            visualize.display_instances(image, rois, masks, class_ids, 'nucleus', scores, show_bbox=False,
                                        show_mask=False, title="Predictions")
            plt.savefig(vis_path)

            # save segmentation result in txt
            # Encode image to RLE. Returns a string of multiple lines
            # source_id = dataset.image_info[image_id]["id"]
            # rle = mask_to_rle(source_id, masks, scores)
            # with open(rle_path, "w") as f:
            #     f.write(rle)

            # save segmentation result to COCO format
            coco_rles = pycocotools.mask.encode(np.asfortranarray(masks.astype('uint8')))
            for k, coco_rle in enumerate(coco_rles):
                coco_seg_result = {'image_id': int(i), 'category_id': int(1), 'segmentation': coco_rle, 'score': float(scores[k])}
                coco_seg_result['segmentation']['counts'] = coco_rle['counts'].decode('utf-8')

                str_ = json.dumps(coco_seg_result, indent=None)
                if len(str_) > 0:
                    output.write(str_)

                # Add comma separator
                output.write(',')

                # Add line break
                output.write('\n')

        # Annotation end
        output.write(']')


if __name__ == '__main__':
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Mask R-CNN for nuclei counting and segmentation')
    parser.add_argument("command",
                        metavar="<command>",
                        default='segment',
                        help="'train', 'detect' or 'segment")
    parser.add_argument('--dataset', required=False,
                        metavar="/path/to/dataset/",
                        default=DIR_DATA_SCIENCE_BOWL_2018,
                        help='Root directory of the dataset')
    parser.add_argument('--weights', required=False,
                        metavar="/path/to/weights.h5",
                        # default='imagenet',     # defautl imagenet weights
                        # default=r'Y:\Users\Jie\CerCyt\nucleus_mrcnn\logs\nucleus20190709T1625-all_kaggle-heads_only\mask_rcnn_nucleus_0020.h5',
                        # default=r'Y:\Users\Jie\CerCyt\nucleus_mrcnn\logs\nucleus20190709T1625-all_kaggle-all_layers\mask_rcnn_nucleus_0040.h5',
                        default=r'Y:\Users\Jie\CerCyt\nucleus_mrcnn\logs\nucleus20190709T1625-16_kaggle_158_step\mask_rcnn_nucleus_0061.h5',
                        # default=KAGGLE_16_WEIGHTS_PATH,
                        help="Path to model weights .h5 file or 'coco'")
    parser.add_argument('--logs', required=False,
                        metavar="/path/to/logs/",
                        default=DEFAULT_LOGS_DIR,
                        help='Logs and checkpoints directory (default=logs/)')
    parser.add_argument('--subset', required=False,
                        metavar="Dataset sub-directory",
                        # default='train',    # stage1_train minus validation set
                        default='stage1_train_cervical_train',
                        help="Subset of dataset to run prediction on")
    parser.add_argument('--image_folder', required=False,
                        metavar="/path/to/image_folder",
                        # default=DEFAULT_IMAGE_DIR,
                        default=r'Y:\Users\Jie\CerCyt\BD_cytology_25\nuclei_segmentation\normal-abnormal',
                        help="The folder path of the images to be segmented")
    args = parser.parse_args()

    # Validate arguments
    if args.command == "train":
        assert args.dataset, "Argument --dataset is required for training"
    elif args.command == "detect":
        assert args.subset, "Provide --subset to run prediction on"
    elif args.command == "segment":
        assert args.image_folder, "Provide --image_folder to run prediction on"

    print("Weights: ", args.weights)
    print("Dataset: ", args.dataset)
    if args.subset:
        print("Subset: ", args.subset)
    print("Logs: ", args.logs)

    # Configurations
    if args.command == "train":
        config = NucleusConfig()
        # if args.subset == 'stage1_train_cervical_train':
        #     config.IMAGES_PER_GPU = 1
        #     config.BATCH_SIZE = 1
        #     config.LEARNING_RATE = 0.000001
        #     config.STEPS_PER_EPOCH = 11    # We have totally 16 cervical nuclei images in Kaggle dataset
        #     config.VALIDATION_STEPS = 5    # Use 11 of them for training and 5 for validation
    else:
        config = NucleusInferenceConfig()
    config.display()

    # Create model
    if args.command == "train":
        model = modellib.MaskRCNN(mode="training", config=config, model_dir=args.logs)
    elif args.command == 'detect':
        model = modellib.MaskRCNN(mode="inference", config=config, model_dir=args.logs)
    else:
        model = modellib.MaskRCNN(mode="inference", config=config, model_dir=args.logs)

# Select weights file to load
    if args.weights.lower() == "coco":
        weights_path = COCO_WEIGHTS_PATH
        # Download weights file
        if not os.path.exists(weights_path):
            utils.download_trained_weights(weights_path)
    elif args.weights.lower() == "last":
        # Find last trained weights
        weights_path = model.find_last()
    elif args.weights.lower() == "imagenet":
        # Start from ImageNet trained weights
        weights_path = model.get_imagenet_weights()
    else:
        weights_path = args.weights

    # Load weights
    print("Loading weights ", weights_path)
    if args.weights.lower() == "coco":
        # Exclude the last layers because they require a matching
        # number of classes
        model.load_weights(weights_path, by_name=True, exclude=[
            "mrcnn_class_logits", "mrcnn_bbox_fc",
            "mrcnn_bbox", "mrcnn_mask"])
    else:
        model.load_weights(weights_path, by_name=True)

    # Train or evaluate
    if args.command == "train":
        train(model, args.dataset, args.subset)
    elif args.command == "detect":
        detect(model, args.dataset, args.subset)
    elif args.command == 'segment':
        segment(model, args.image_folder)
    else:
        print("'{}' is not recognized. "
              "Use 'train' or 'detect'".format(args.command))
