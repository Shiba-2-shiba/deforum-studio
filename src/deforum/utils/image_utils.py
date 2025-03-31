import gc
import os
import re
import socket
import time
from threading import Thread

import PIL.Image
import cv2
import numpy as np
import requests
import torch # For unsharp_mask and potentially TF adjustments
import torchvision.transforms.functional as TF # For potential adjustments
from PIL import (Image, ImageChops, ImageOps)

from scipy.ndimage import gaussian_filter # For unsharp_mask (Gaussian Blur part, though cv2 might be used in torch version indirectly)
try:
    from skimage.exposure import match_histograms
except ImportError:
    print("[Deforum] Warning: scikit-image not available. match_histograms will not work if called directly.")
    match_histograms = None # Define as None if skimage not available

# Assuming these util locations are correct relative to where image_utils.py is
try:
    from .deforum_word_masking_util import get_word_mask
    from .video_frame_utils import get_frame_name
    from .gradio_utils import clean_gradio_path_strings # Check if this util exists/is needed in ComfyUI context
    from .logging_config import logger
except ImportError as e:
     print(f"[Deforum] Warning: Could not import some local utilities in image_utils.py: {e}")
     # Define dummy functions or handle missing imports if necessary
     def get_word_mask(root, frame_image, content): return Image.new('1', frame_image.size, 0) # Dummy
     def get_frame_name(path): return "frame" # Dummy
     def clean_gradio_path_strings(p): return p # Dummy
     class DummyLogger:
         def info(self, msg): print(f"INFO: {msg}")
         def warning(self, msg): print(f"WARN: {msg}")
         def error(self, msg): print(f"ERROR: {msg}")
     logger = DummyLogger() # Dummy


DEBUG_MODE = True


# --- New function from A1111 Deforum ---
def optimized_pixel_diffusion_blend(image1, image2, alpha, cc_mix_outdir=None, timestring=None, idx=None):
    """
    Blends image1 onto image2 using random pixel selection based on alpha.
    alpha = 1.0 means 100% image1 pixels are kept where mask allows.
    alpha = 0.0 means 100% image2 pixels are kept where mask allows.
    Assumes image1 and image2 are NumPy arrays (e.g., BGR uint8).
    """
    alpha = min(max(alpha, 0), 1)
    beta = 1 - alpha # Proportion for image2

    # Ensure inputs are numpy arrays
    if not isinstance(image1, np.ndarray):
        image1 = np.array(image1)
    if not isinstance(image2, np.ndarray):
        image2 = np.array(image2)

    if image1.shape != image2.shape:
        logger.warning(f"Shape mismatch in optimized_pixel_diffusion_blend: {image1.shape} vs {image2.shape}. Resizing image1.")
        try:
            image1 = cv2.resize(image1, (image2.shape[1], image2.shape[0]), interpolation=cv2.INTER_AREA)
        except Exception as e:
            logger.error(f"Failed to resize image1 in blend: {e}. Returning image2.")
            return image2

    # Create a random matrix the same shape as your image height/width
    # Ensure it matches the potentially C-contiguous array from cv2
    h, w = image1.shape[:2]
    random_matrix = np.random.uniform(0, 1, (h, w))

    # Create masks based on alpha
    alpha_mask = random_matrix < alpha

    # Initialize result as a copy of image2 (the base image)
    # Ensure contiguous array for potentially faster assignment
    result = np.copy(image2)

    # Apply the mask: Where alpha_mask is True, copy pixels from image1
    # Using np.where for potentially better performance on boolean indexing
    # result[alpha_mask] = image1[alpha_mask] # Direct boolean indexing
    # Alternatively using np.where:
    # This requires broadcasting image1 and image2 correctly if they have color channels
    if result.ndim == 3 and image1.ndim == 3:
        alpha_mask_3d = np.stack([alpha_mask]*result.shape[2], axis=-1)
        result = np.where(alpha_mask_3d, image1, result)
    elif result.ndim == 2 and image1.ndim == 2: # Grayscale case
        result = np.where(alpha_mask, image1, result)
    else: # Fallback or handle error if dimensions mismatch unexpectedly
         logger.warning("Unexpected dimensions in optimized_pixel_diffusion_blend after mask creation. Using boolean indexing fallback.")
         result[alpha_mask] = image1[alpha_mask] # Fallback


    # Debug/Optional: Save the intermediate blended image
    if cc_mix_outdir is not None and timestring is not None and idx is not None:
        try:
            if not os.path.exists(cc_mix_outdir):
                os.makedirs(cc_mix_outdir)
            full_filepath = os.path.join(cc_mix_outdir, f'{timestring}_{idx:09}_cc_blend.jpg')
            # Ensure result is uint8 for saving
            cv2.imwrite(full_filepath, result.astype(np.uint8))
        except Exception as e:
            logger.error(f"Error saving cc_mix debug image: {e}")

    return result
# --- End of new function ---


# --- Original maintain_colors (Commented out as logic moved to DeforumColorMatchNode) ---
def maintain_colors(prev_img, color_match_sample, mode):
    """ Original function, kept for compatibility even if not directly used by the node now. """
    # is_skimage_v20_or_higher = True # Assume modern version or handle check differently
    # Check skimage availability (added check)
    if match_histograms is None:
         logger.warning("scikit-image not available for maintain_colors.")
         return prev_img # Return original if skimage missing

    # Check skimage version for kwargs (using self.match_histograms_kwargs might not work here as it's not in a class)
    # Let's try a simplified check or assume default args
    match_histograms_kwargs = {'channel_axis': -1} # Assume v0.20+ default
    try:
         # Optional: Add version check here if needed, otherwise use default above
         pass
    except Exception:
         match_histograms_kwargs = {'multichannel': True} # Fallback

    try:
        # Ensure inputs are BGR numpy arrays uint8
        if not isinstance(prev_img, np.ndarray): prev_img = np.array(prev_img)
        if not isinstance(color_match_sample, np.ndarray): color_match_sample = np.array(color_match_sample)

        # Ensure uint8 type
        prev_img_u8 = prev_img.astype(np.uint8)
        color_match_sample_u8 = color_match_sample.astype(np.uint8)

        # Ensure shapes match (added check)
        if prev_img_u8.shape != color_match_sample_u8.shape:
             logger.warning(f"maintain_colors: Shape mismatch {prev_img_u8.shape} vs {color_match_sample_u8.shape}. Resizing sample.")
             color_match_sample_u8 = cv2.resize(color_match_sample_u8, (prev_img_u8.shape[1], prev_img_u8.shape[0]), interpolation=cv2.INTER_AREA)


        if mode == 'RGB':
             prev_img_rgb = cv2.cvtColor(prev_img_u8, cv2.COLOR_BGR2RGB)
             color_match_rgb = cv2.cvtColor(color_match_sample_u8, cv2.COLOR_BGR2RGB) # Assume sample needs conversion too
             matched = match_histograms(prev_img_rgb, color_match_rgb, **match_histograms_kwargs)
             return cv2.cvtColor(matched.astype(np.uint8), cv2.COLOR_RGB2BGR)
        elif mode == 'HSV':
             prev_img_hsv = cv2.cvtColor(prev_img_u8, cv2.COLOR_BGR2HSV)
             color_match_hsv = cv2.cvtColor(color_match_sample_u8, cv2.COLOR_BGR2HSV)
             matched_hsv = match_histograms(prev_img_hsv, color_match_hsv, **match_histograms_kwargs)
             return cv2.cvtColor(matched_hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
        else:  # LAB (Default)
             prev_img_lab = cv2.cvtColor(prev_img_u8, cv2.COLOR_BGR2LAB)
             color_match_lab = cv2.cvtColor(color_match_sample_u8, cv2.COLOR_BGR2LAB)
             matched_lab = match_histograms(prev_img_lab, color_match_lab, **match_histograms_kwargs)
             return cv2.cvtColor(matched_lab.astype(np.uint8), cv2.COLOR_LAB2BGR)
    except Exception as e:
         logger.error(f"Error in original maintain_colors (mode: {mode}): {e}")
         return prev_img # Return original on error
# --- ここまでコメント解除 ---

 --- IMAGE FUNCTIONS (Original functions below, plus modified unsharp_mask) ---

def load_image(image_path: str):
    if isinstance(image_path, str):
        image_path = clean_gradio_path_strings(image_path) # Keep if needed, might do nothing in comfy
        if image_path.startswith('http://') or image_path.startswith('https://'):
            logger.info(f"Attempting to load image from URL: {image_path}")
            # Check internet connection (simple check)
            try:
                host = socket.gethostbyname("www.google.com") # Or a more reliable host
                s = socket.create_connection((host, 80), 2)
                s.close()
                logger.info("Internet connection check successful.")
            except Exception as e:
                 logger.error(f"No active internet connection detected: {e}")
                 raise ConnectionError(
                     "There is no active internet connection available - please use local masks and init files only.")

            # Attempt to download
            try:
                response = requests.get(image_path, stream=True, timeout=10) # Added timeout
                response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
            except requests.exceptions.RequestException as e:
                logger.error(f"Failed to download image: {e}")
                raise ConnectionError(f"Failed to download image due to connection error or invalid URL: {e}")

            logger.info(f"Successfully connected to URL. Status code: {response.status_code}")
            try:
                 image = Image.open(response.raw).convert('RGB')
            except Exception as e:
                 logger.error(f"Failed to open image from downloaded response: {e}")
                 raise RuntimeError(f"Could not open image from URL {image_path}: {e}")

        else: # Local file path
            logger.info(f"Attempting to load image from local path: {image_path}")
            if not os.path.exists(image_path):
                 logger.error(f"Image path does not exist: {image_path}")
                 raise FileNotFoundError(f"Init image path or mask image path is not valid: {image_path}")
            try:
                 image = Image.open(image_path).convert('RGB')
            except Exception as e:
                 logger.error(f"Failed to open local image file: {e}")
                 raise RuntimeError(f"Could not open image from local path {image_path}: {e}")
        return image
    elif isinstance(image_path, PIL.Image.Image):
        logger.info("Input is already a PIL image.")
        # Ensure it's RGB
        if image_path.mode != 'RGB':
            logger.warning(f"Converting PIL image from mode {image_path.mode} to RGB.")
            return image_path.convert('RGB')
        return image_path
    else:
        logger.error(f"Unsupported image_path type: {type(image_path)}")
        raise TypeError(f"load_image expects a string path or PIL image, got {type(image_path)}")


def blank_if_none(mask, w, h, mode):
    return Image.new(mode, (w, h), 0) if mask is None else mask


def none_if_blank(mask):
    if mask is None: return None
    try:
        extrema = mask.getextrema()
        # Check for single band (L, 1) or multi-band images where all bands are blank
        if isinstance(extrema, tuple): # Single band
            if extrema == (0, 0): return None
        elif isinstance(extrema, list): # Multi-band
            if all(ex == (0, 0) for ex in extrema): return None
    except Exception as e:
        logger.warning(f"Could not get extrema for mask check: {e}")
    return mask


def get_resized_image_from_filename(im_path, dimensions):
    img = cv2.imread(im_path)
    if img is None:
        logger.error(f"Could not read image file: {im_path}")
        return None
    return cv2.resize(img, (dimensions[0], dimensions[1]), cv2.INTER_AREA)


def center_crop_image(img, w, h):
    y, x = img.shape[:2] # Works for color and grayscale
    start_x = max(0, x // 2 - w // 2)
    start_y = max(0, y // 2 - h // 2)
    end_x = min(x, start_x + w)
    end_y = min(y, start_y + h)
    cropped_img = img[start_y:end_y, start_x:end_x]
    # Optional: Pad if crop is smaller than target (e.g., if original image was smaller)
    # This might be needed depending on how it's used.
    # Example padding:
    # if cropped_img.shape[1] < w or cropped_img.shape[0] < h:
    #     pad_x = (w - cropped_img.shape[1]) // 2
    #     pad_y = (h - cropped_img.shape[0]) // 2
    #     pad_x_rem = w - cropped_img.shape[1] - pad_x
    #     pad_y_rem = h - cropped_img.shape[0] - pad_y
    #     cropped_img = cv2.copyMakeBorder(cropped_img, pad_y, pad_y_rem, pad_x, pad_x_rem, cv2.BORDER_CONSTANT, value=[0,0,0]) # Adjust value as needed
    return cropped_img


def autocontrast_grayscale(image, low_cutoff=0.0, high_cutoff=100.0):
    """ Performs autocontrast on a grayscale NumPy array image. """
    if image.ndim != 2:
        logger.warning("autocontrast_grayscale expects a 2D grayscale image.")
        return image # Return original if not grayscale
    try:
        min_val = np.percentile(image, low_cutoff)
        max_val = np.percentile(image, high_cutoff)

        if max_val == min_val: # Avoid division by zero if image is flat
            logger.info("Image has zero contrast range.")
            # Return 0, 128, or 255 depending on preference for flat images
            return np.full_like(image, 128, dtype=np.uint8)

        # Scale the image
        # Use floating point for calculations to avoid precision issues
        image_float = image.astype(np.float32)
        scaled_image = 255.0 * (image_float - min_val) / (max_val - min_val)

        # Clip values and convert back to uint8
        image_uint8 = np.clip(scaled_image, 0, 255).astype(np.uint8)
        return image_uint8
    except Exception as e:
        logger.error(f"Error during autocontrast_grayscale: {e}")
        return image # Return original on error


def image_transform_ransac(image_cv2, m, hybrid_motion, depth=None):
    # RANSAC usually finds a perspective or affine matrix 'm'
    # This function just applies it. The RANSAC calculation happens elsewhere.
    if hybrid_motion == "Perspective":
        return image_transform_perspective(image_cv2, m, depth)
    else:  # Affine or other (assume Affine as fallback)
        return image_transform_affine(image_cv2, m, depth)


def image_transform_optical_flow(img, flow, flow_factor):
    """ Applies optical flow warping to an image. Assumes flow is [H, W, 2] """
    if img is None or flow is None:
        logger.warning("image_transform_optical_flow received None input.")
        return img

    h, w = img.shape[:2]
    flow_h, flow_w = flow.shape[:2]

    if h != flow_h or w != flow_w:
         logger.warning(f"Image shape {img.shape[:2]} and flow shape {flow.shape[:2]} mismatch. Resizing flow.")
         flow = cv2.resize(flow, (w, h), interpolation=cv2.INTER_LINEAR)
         # Resizing flow might require scaling the flow vectors too, this depends on the algorithm.
         # Simple resize might be incorrect. Assuming flow vectors scale with resize for now.
         flow[:,:,0] *= (w / flow_w)
         flow[:,:,1] *= (h / flow_h)


    # Apply flow factor
    if flow_factor != 1.0:
        flow = flow * flow_factor

    # Flow represents where each pixel *comes from*.
    # Remap needs the absolute source coordinate for each destination pixel.
    # Create grid of destination coordinates
    grid_x, grid_y = np.meshgrid(np.arange(w), np.arange(h))
    map_x = grid_x.astype(np.float32) + flow[:, :, 0]
    map_y = grid_y.astype(np.float32) + flow[:, :, 1]

    # Perform the remapping
    # cv2.BORDER_REFLECT_101 is often good for seamless borders
    remapped_img = cv2.remap(img, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)

    return remapped_img


def image_transform_affine(image_cv2, m, depth=None):
    if depth is not None:
         logger.warning("Depth-based affine warp not implemented. Using standard warp.")
         # Implement depth-based warp here if needed
         # return depth_based_affine_warp(image_cv2, depth, m)

    if m is None:
        logger.warning("Affine matrix 'm' is None. Returning original image.")
        return image_cv2

    # Standard affine warp
    return cv2.warpAffine(
        image_cv2,
        m, # Should be a 2x3 matrix
        (image_cv2.shape[1], image_cv2.shape[0]), # (width, height)
        flags=cv2.INTER_LINEAR, # Linear interpolation is common
        borderMode=cv2.BORDER_REFLECT_101 # Reflect pixels at border
    )


def image_transform_perspective(image_cv2, m, depth=None):
    if depth is not None:
         logger.warning("Depth-based perspective warp not implemented. Using standard warp.")
         # Implement 3D perspective render here if needed
         # return render_3d_perspective(image_cv2, depth, m)

    if m is None:
        logger.warning("Perspective matrix 'm' is None. Returning original image.")
        return image_cv2

    # Standard perspective warp
    return cv2.warpPerspective(
        image_cv2,
        m, # Should be a 3x3 matrix
        (image_cv2.shape[1], image_cv2.shape[0]), # (width, height)
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101
    )
    
# image_utils.py に以下の関数定義を追加します
# 他の画像処理関数 (unsharp_mask など) の近くに追加すると良いでしょう。

def custom_gaussian_blur(input_array, blur_size, sigma):
    """ Applies Gaussian blur using scipy.ndimage.gaussian_filter. """
    # Ensure sigma is appropriate for the filter dimensions
    # sigma=(sigma, sigma, 0) assumes a 3D array (H, W, C) where blur is only applied spatially.
    # If input_array could be 2D (grayscale), this might need adjustment.
    try:
        if input_array.ndim == 3:
            sigma_tuple = (sigma, sigma, 0)
        elif input_array.ndim == 2:
            sigma_tuple = (sigma, sigma)
        else:
            logger.error(f"Unsupported array dimension {input_array.ndim} for custom_gaussian_blur.")
            return input_array # Return original on unsupported dimension

        # truncate parameter relates kernel size to sigma. Default is 4.0
        # Using blur_size directly might not be the intended use of truncate.
        # Let's use the default truncate value unless blur_size is specifically meant for it.
        # truncate=4.0 is standard. If blur_size is intended as kernel pixel size, it's different.
        # Assuming blur_size here might be intended as the truncate factor for compatibility.
        truncate_value = float(blur_size) if blur_size is not None else 4.0

        return gaussian_filter(input_array,
                               sigma=sigma_tuple,
                               order=0, # order=0 means Gaussian smoothing
                               mode='constant', # How borders are handled
                               cval=0.0, # Value for constant border
                               truncate=truncate_value) # Cut off filter at truncate*sigma standard deviations
    except Exception as e:
        logger.error(f"Error during custom_gaussian_blur: {e}")
        return input_array # Return original on error


# MASK FUNCTIONS

def load_image_with_mask(path: str, shape=None, use_alpha_as_mask=False):
    # Load the base image (using our robust load_image function)
    try:
        image = load_image(path)
    except (FileNotFoundError, ConnectionError, RuntimeError, TypeError) as e:
        logger.error(f"Failed to load base image in load_image_with_mask: {e}")
        return None, None # Return None for both if base image fails

    mask_image = None

    if use_alpha_as_mask:
        # Try reloading with alpha channel if PIL image supports it
        # Note: load_image currently forces RGB, so need to reload or handle differently
        logger.info("Trying to reload image to get alpha channel for mask.")
        try:
             # Reload specifically asking for RGBA if possible
             if isinstance(path, str):
                 if not path.startswith('http'): # Reload local file if possible
                     img_with_alpha = Image.open(path)
                     if img_with_alpha.mode == 'RGBA':
                         logger.info("Successfully loaded RGBA image.")
                         image = img_with_alpha.convert('RGB') # Keep the RGB version loaded earlier or convert again
                         mask_image = img_with_alpha.split()[-1].convert('L')
                     else:
                         logger.warning(f"Image at {path} does not have an alpha channel (mode: {img_with_alpha.mode}). Cannot use alpha as mask.")
                 else:
                      logger.warning("Cannot reliably reload URL image to check for alpha channel. Ignoring use_alpha_as_mask for URL.")
             elif isinstance(path, PIL.Image.Image):
                  if path.mode == 'RGBA':
                      logger.info("Input PIL image has alpha channel.")
                      image = path.convert('RGB')
                      mask_image = path.split()[-1].convert('L')
                  else:
                      logger.warning(f"Input PIL image does not have alpha channel (mode: {path.mode}). Cannot use alpha as mask.")

        except Exception as e:
             logger.error(f"Error trying to load/access alpha channel: {e}")

    # Resize image and mask if shape is provided
    if shape is not None:
        logger.info(f"Resizing image to {shape}.")
        image = image.resize(shape, resample=Image.LANCZOS)
        if mask_image is not None:
            logger.info(f"Resizing mask to {shape}.")
            mask_image = mask_image.resize(shape, resample=Image.LANCZOS) # Use LANCZOS for mask too? Maybe NEAREST is better?

    # Check if mask is blank after potential loading/resizing
    if mask_image is not None:
         mask_image = none_if_blank(mask_image)
         if mask_image is None:
              logger.info("Mask derived from alpha channel is blank. Discarding mask.")


    return image, mask_image


def prepare_mask(mask_input, mask_shape, mask_brightness_adjust=1.0, mask_contrast_adjust=1.0):
    """ Prepares mask for use: loads, resizes, adjusts, converts to L. """
    if mask_input is None:
        logger.warning("prepare_mask received None input.")
        # Return a default blank mask matching the shape
        return Image.new('L', mask_shape, 0)

    try:
        if isinstance(mask_input, Image.Image):
            mask = mask_input
        else: # Assume string path
            mask = load_image(mask_input) # load_image returns RGB, need L
    except Exception as e:
        logger.error(f"Failed to load mask input '{mask_input}': {e}")
        # Return a default blank mask on load failure
        return Image.new('L', mask_shape, 0)

    # Convert to Grayscale ('L') before adjustments
    if mask.mode != 'L':
        mask = mask.convert('L')

    # Resize
    if mask.size != mask_shape:
        mask = mask.resize(mask_shape, resample=Image.LANCZOS) # Or Image.NEAREST for masks?

    # Adjustments (Using PIL ImageOps for simplicity, TF requires tensor conversion)
    # Brightness/Contrast on masks can be tricky. Usually thresholding is preferred.
    # Applying these adjustments might make binary masks non-binary.
    # Consider if simple thresholding is better than brightness/contrast.
    if mask_brightness_adjust != 1.0:
        logger.warning("Applying brightness adjustment to mask. This might yield non-binary results.")
        # PIL Brightness: factor > 1 increases, < 1 decreases
        enhancer = ImageEnhance.Brightness(mask)
        mask = enhancer.enhance(mask_brightness_adjust)

    if mask_contrast_adjust != 1.0:
        logger.warning("Applying contrast adjustment to mask. This might yield non-binary results.")
        # PIL Contrast: factor > 1 increases, < 1 decreases
        enhancer = ImageEnhance.Contrast(mask)
        mask = enhancer.enhance(mask_contrast_adjust)

    # Ensure it's still 'L' mode after potential enhancements
    if mask.mode != 'L':
        mask = mask.convert('L')

    # Optional: Binarize the mask after adjustments if needed
    # threshold_value = 128
    # mask = mask.point(lambda p: 255 if p > threshold_value else 0, 'L')

    return mask

# Check mask function (simplified - primarily checks if it's completely black)
# The original comment mentioned issues with all-black masks being rejected.
# This version logs if it's all black but doesn't reject it outright.
def check_mask_for_errors(mask_input, invert_mask=False):
    if mask_input is None:
        logger.info("Mask check: Input is None.")
        return None
    try:
        extrema = mask_input.getextrema()
        is_blank = False
        if isinstance(extrema, tuple): # L mode
            if extrema == (0, 0): is_blank = True
            if extrema == (255, 255): is_blank = True # Also check if all white
        # Add checks for other modes if necessary

        if invert_mask:
            if extrema == (255, 255): # If all white, inverting makes it all black
                 logger.info("Mask check: Mask is all white and will be inverted to all black.")
            elif extrema == (0,0): # If all black, inverting makes it all white
                 logger.info("Mask check: Mask is all black and will be inverted to all white.")
        else:
            if is_blank:
                logger.info("Mask check: Mask is entirely black (or white).")

    except Exception as e:
        logger.warning(f"Could not get extrema for mask check: {e}")

    # Return the mask regardless, let downstream decide how to handle blank masks
    return mask_input


def get_mask(args):
    """ Helper to get mask based on args object """
    mask_file = getattr(args, 'mask_file', None)
    width = getattr(args, 'width', 512)
    height = getattr(args, 'height', 512)
    mask_brightness = getattr(args, 'mask_brightness_adjust', 1.0)
    mask_contrast = getattr(args, 'mask_contrast_adjust', 1.0)
    invert = getattr(args, 'invert_mask', False) # Needed for check_mask

    if mask_file is None:
        logger.info("No mask_file specified in args.")
        return None

    prepared_mask = prepare_mask(mask_file, (width, height), mask_brightness, mask_contrast)
    # checked_mask = check_mask_for_errors(prepared_mask, invert) # Check but don't reject
    # return checked_mask
    return prepared_mask # Return prepared mask, let caller handle inversion/checking if needed


def get_mask_from_file(mask_file, args):
    """ Helper to get mask from a specific file path using args for settings """
    width = getattr(args, 'width', 512)
    height = getattr(args, 'height', 512)
    mask_brightness = getattr(args, 'mask_brightness_adjust', 1.0)
    mask_contrast = getattr(args, 'mask_contrast_adjust', 1.0)
    invert = getattr(args, 'invert_mask', False) # Needed for check_mask

    if mask_file is None or not isinstance(mask_file, str):
        logger.warning(f"Invalid mask_file path: {mask_file}")
        return None

    prepared_mask = prepare_mask(mask_file, (width, height), mask_brightness, mask_contrast)
    # checked_mask = check_mask_for_errors(prepared_mask, invert)
    # return checked_mask
    return prepared_mask


# --- Unsharp Mask (Using Torch version from previous response) ---
from PIL import ImageEnhance # Needed for prepare_mask adjustments

def unsharp_mask(img, kernel_size=(5, 5), sigma=1.0, amount=1.0, threshold=0, mask=None):
    """ Applies unsharp mask using PyTorch for potential GPU acceleration. """
    if amount == 0:
        return img

    if not isinstance(img, np.ndarray):
         logger.warning("unsharp_mask expects a NumPy array input.")
         # Try to convert if PIL image
         if isinstance(img, Image.Image):
              img = np.array(img.convert('RGB')) # Assume RGB for conversion
              if img is None: return None # Failed conversion
         else:
             return img # Return original if not numpy or PIL

    # Determine device
    # Avoid calling cuda.is_available frequently if possible
    # Maybe pass device as arg or determine once globally?
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Ensure image is uint8 before converting to float tensor
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)


    # Convert NumPy image (H, W, C) [assumed BGR or RGB] to Tensor (C, H, W) float
    # Handle potential grayscale (H, W) input
    if img.ndim == 3:
        img_tensor = torch.from_numpy(img).permute(2, 0, 1).float().to(device) / 255.0
    elif img.ndim == 2:
        img_tensor = torch.from_numpy(img).unsqueeze(0).float().to(device) / 255.0 # Add channel dim
    else:
        logger.error(f"Unsupported image dimensions for unsharp_mask: {img.ndim}")
        return img

    channels, height, width = img_tensor.shape

    # Gaussian Blur using TorchVision transforms
    # kernel_size needs to be odd
    k_h = kernel_size[0] if kernel_size[0] % 2 != 0 else kernel_size[0] + 1
    k_w = kernel_size[1] if kernel_size[1] % 2 != 0 else kernel_size[1] + 1
    gaussian_blur = TF.GaussianBlur(kernel_size=(k_h, k_w), sigma=sigma)
    blurred_tensor = gaussian_blur(img_tensor)

    # Calculate sharpened tensor
    # amount = 1.0 means (2 * img - 1 * blurred)
    sharpened_tensor = (amount + 1.0) * img_tensor - amount * blurred_tensor

    # Apply threshold
    if threshold > 0:
        # Threshold is typically applied in 0-255 range, adjust here for 0-1 range
        threshold_norm = threshold / 255.0
        low_contrast_mask = torch.abs(img_tensor - blurred_tensor) < threshold_norm
        # Apply mask across channels if needed
        if low_contrast_mask.shape[0] == 1 and channels > 1:
            low_contrast_mask = low_contrast_mask.repeat(channels, 1, 1)
        sharpened_tensor = torch.where(low_contrast_mask, img_tensor, sharpened_tensor)

    # Apply mask if provided
    if mask is not None:
        try:
            # Ensure mask is a PIL image, L mode, matching size
            if not isinstance(mask, Image.Image): mask = Image.fromarray(mask) # Assume numpy if not PIL
            if mask.mode != 'L': mask = mask.convert('L')
            if mask.size != (width, height): mask = mask.resize((width, height), resample=Image.NEAREST)

            # Convert mask to tensor (1, H, W), normalize to 0-1
            mask_tensor = TF.to_tensor(mask).to(device) # Should already be [0, 1] range

            # Ensure mask_tensor has same number of channels for broadcasting if needed
            # if mask_tensor.shape[0] == 1 and sharpened_tensor.shape[0] > 1:
            #     mask_tensor = mask_tensor.repeat(sharpened_tensor.shape[0], 1, 1)

            # Blend: sharpened * mask + original * (1 - mask)
            sharpened_tensor = sharpened_tensor * mask_tensor + img_tensor * (1.0 - mask_tensor)

        except Exception as e:
            logger.error(f"Failed to apply mask in unsharp_mask: {e}")


    # Clamp, convert back to numpy uint8
    sharpened_tensor = torch.clamp(sharpened_tensor * 255.0, 0, 255).byte()

    # Convert back to NumPy (H, W, C) or (H, W)
    if sharpened_tensor.shape[0] == 3: # Color
         output_img = sharpened_tensor.cpu().numpy().transpose(1, 2, 0)
    elif sharpened_tensor.shape[0] == 1: # Grayscale
         output_img = sharpened_tensor.cpu().numpy().squeeze(0)
    else: # Should not happen
         logger.error("Unexpected tensor shape after unsharp mask.")
         return img

    return output_img

# --- Rest of the original functions ---

def do_overlay_mask(args, anim_args, img, frame_idx, is_bgr_array=False):
    """ Overlays init/video frame onto img using mask """
    current_mask = None
    current_frame = None
    img_pil = None # Work with PIL images internally

    # Convert input img to PIL RGB if it's not already
    if isinstance(img, Image.Image):
        img_pil = img.convert('RGB') if img.mode != 'RGB' else img
    elif isinstance(img, np.ndarray):
        logger.info("Overlay: Input is NumPy array, converting to PIL.")
        if is_bgr_array: # Input is BGR
            img_pil = Image.fromarray(cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2RGB))
        else: # Assume input is RGB
            img_pil = Image.fromarray(img.astype(np.uint8))
    else:
        logger.error("Overlay: Unsupported input image type.")
        return img # Return original if type is wrong

    img_w, img_h = img_pil.size

    # --- Load Mask ---
    mask_source = None
    if anim_args.use_mask_video:
        mask_vid_path = getattr(anim_args, 'video_mask_path', None)
        if mask_vid_path:
            # Need function to get specific frame path from video path/folder
            # Assuming maskframes are saved in args.outdir/maskframes/
            mask_frame_name = get_frame_name(mask_vid_path) # Util function
            mask_source = os.path.join(args.outdir, 'maskframes', mask_frame_name + f"{frame_idx:09}.jpg")
            logger.info(f"Overlay: Loading mask frame: {mask_source}")
        else: logger.warning("Overlay: use_mask_video is True, but video_mask_path is missing.")
    elif args.use_mask:
        mask_source = getattr(args, 'mask_file', None)
        logger.info(f"Overlay: Loading static mask file: {mask_source}")
        # mask_image is sometimes preloaded in args? Check that.
        if hasattr(args, 'mask_image') and args.mask_image is not None:
             logger.info("Overlay: Using pre-loaded args.mask_image")
             current_mask = args.mask_image # Assume it's a PIL image
             mask_source = None # Don't load from file if pre-loaded exists

    if mask_source:
        try:
            current_mask = load_image(mask_source) # Load as RGB first
        except Exception as e:
            logger.error(f"Overlay: Failed to load mask from {mask_source}: {e}")
            current_mask = None

    # --- Load Frame to Overlay ---
    overlay_frame_source = None
    if anim_args.use_mask_video: # If using video mask, use corresponding video init frame
         init_vid_path = getattr(anim_args, 'video_init_path', None)
         if init_vid_path:
              init_frame_name = get_frame_name(init_vid_path)
              overlay_frame_source = os.path.join(args.outdir, 'inputframes', init_frame_name + f"{frame_idx:09}.jpg")
              logger.info(f"Overlay: Loading init frame: {overlay_frame_source}")
         else: logger.warning("Overlay: use_mask_video is True, but video_init_path is missing.")
    elif args.use_mask: # If using static mask
         overlay_frame_source = getattr(args, 'init_image', None)
         if overlay_frame_source:
             logger.info(f"Overlay: Loading static init image: {overlay_frame_source}")
         else: # If no init_image, overlay the input image itself (no change unless mask is inverted?)
              logger.info("Overlay: No init_image specified, using input image for overlay (no effect unless mask inverted).")
              current_frame = img_pil # Use the input image itself

    if overlay_frame_source:
        try:
            current_frame = load_image(overlay_frame_source)
        except Exception as e:
            logger.error(f"Overlay: Failed to load overlay frame from {overlay_frame_source}: {e}")
            current_frame = None

    # --- Perform Overlay ---
    if current_mask is not None and current_frame is not None:
        try:
            # Prepare mask: Resize, convert to L, invert if needed
            current_mask = current_mask.resize((img_w, img_h), Image.LANCZOS).convert('L')
            if args.invert_mask:
                current_mask = ImageOps.invert(current_mask)

            # Prepare frame: Resize
            current_frame = current_frame.resize((img_w, img_h), Image.LANCZOS).convert('RGB')

            # Composite: background=img_pil, foreground=current_frame, mask=current_mask
            img_pil = Image.composite(current_frame, img_pil, current_mask)
            logger.info("Overlay: Composite successful.")

        except Exception as e:
             logger.error(f"Overlay: Error during composition: {e}")
             # Return original image on error during composite
             if isinstance(img, Image.Image): return img
             elif isinstance(img, np.ndarray): return img
             else: return None # Should not happen

    elif current_mask is None:
        logger.warning("Overlay: Mask could not be loaded or prepared. Skipping overlay.")
    elif current_frame is None:
        logger.warning("Overlay: Frame to overlay could not be loaded or prepared. Skipping overlay.")


    # Return in the original format if possible
    if isinstance(img, Image.Image):
        return img_pil
    elif isinstance(img, np.ndarray):
        if is_bgr_array:
            # Convert final PIL RGB back to NumPy BGR
            return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
        else:
            # Convert final PIL RGB back to NumPy RGB
            return np.array(img_pil)
    else:
        return img_pil # Return PIL if original type unknown


# Function for composing masks based on string expressions
# This seems complex and relies heavily on args and root object structure
# Might need significant adaptation for ComfyUI context if used directly
def compose_mask(root, args, mask_seq, val_masks, frame_image, inner_idx: int = 0):
    """
    Recursively parses a mask expression string (e.g., "<cat> & [mask.png] | (!{everywhere})" )
    Requires 'root' object for word masks and 'args' for settings.
    'val_masks' is a dict storing intermediate PIL mask results.
    'frame_image' is the PIL image for word masking context.
    Returns the key in val_masks containing the final composed mask.
    """
    if not isinstance(mask_seq, str):
         logger.error("compose_mask: mask_seq must be a string.")
         return None

    logger.info(f"Composing mask: {mask_seq} (Inner index: {inner_idx})")

    # Predefined masks often available in val_masks
    width = getattr(args, 'width', 512)
    height = getattr(args, 'height', 512)
    if 'everywhere' not in val_masks:
         val_masks['everywhere'] = Image.new('1', (width, height), 1)
    if 'nowhere' not in val_masks:
          val_masks['nowhere'] = Image.new('1', (width, height), 0)


    # --- Step 1: Recursive Parenthesis Pass ---
    seq = ""
    inner_seq = ""
    parentheses_counter = 0
    for char_idx, c in enumerate(mask_seq):
        if c == '(':
            if parentheses_counter == 0: # Start of a new parenthesis block
                inner_seq = "" # Reset inner sequence
            else:
                inner_seq += c # Part of nested parenthesis
            parentheses_counter += 1
        elif c == ')':
            parentheses_counter -= 1
            if parentheses_counter < 0:
                raise ValueError(f"Mismatched closing parenthesis at index {char_idx} in mask sequence: {mask_seq}")
            if parentheses_counter == 0: # End of a block
                logger.debug(f"Processing parenthesis block: {inner_seq}")
                # Recursively call compose_mask on the inner sequence
                # Pass a unique key prefix or manage inner_idx carefully
                # Need a way to generate unique keys for sub-results
                new_key = f"sub_{inner_idx}_{len(val_masks)}" # Generate a temporary key
                result_key = compose_mask(root, args, inner_seq, val_masks, frame_image, inner_idx + 1) # Recursive call
                if result_key: # If recursion successful
                    # Store the result of the parenthesis block with a placeholder key
                    # This part is tricky - the original code replaces directly,
                    # but tracking keys might be safer. Let's assume result_key is the final key for the sub-expression.
                     seq += f"{{{result_key}}}" # Append placeholder for the evaluated block
                else:
                     logger.error("Recursive mask composition failed for block.")
                     # Handle error, maybe raise exception or return None
                     return None
                inner_seq = "" # Reset inner sequence
            else: # Still inside nested parenthesis
                inner_seq += c
        elif parentheses_counter > 0: # Inside parenthesis block
            inner_seq += c
        else: # Outside parenthesis
            seq += c

    if parentheses_counter != 0:
        raise ValueError(f"Mismatched opening parenthesis in mask sequence: {mask_seq}")

    mask_seq = seq # Sequence with evaluated parenthesis replaced by {key}
    logger.debug(f"Mask sequence after parenthesis pass: {mask_seq}")


    # --- Step 2: Load File and Word Masks ---
    # Replace [...] and <...> with {key} placeholders

    # File masks: [filepath.png]
    pattern_file = r'\[(?P<inner>[^\[\]]+?)\]' # Non-greedy match inside brackets
    def parse_file(match_object):
        nonlocal inner_idx # Use inner_idx passed to function
        content = match_object.group('inner').strip()
        logger.debug(f"Parsing file mask: [{content}]")
        # Use content hash or unique ID for key to potentially reuse masks
        file_key = f"file_{hash(content)}_{width}x{height}"
        if file_key not in val_masks:
             try:
                 mask_img = get_mask_from_file(content, args) # Returns PIL 'L' mask
                 if mask_img:
                     val_masks[file_key] = mask_img.convert('1') # Convert to binary
                 else: # Handle mask loading failure
                      logger.warning(f"Could not load file mask: {content}. Using 'nowhere'.")
                      val_masks[file_key] = val_masks['nowhere']
             except Exception as e:
                  logger.error(f"Error loading file mask [{content}]: {e}. Using 'nowhere'.")
                  val_masks[file_key] = val_masks['nowhere']
        return f"{{{file_key}}}"
    mask_seq = re.sub(pattern_file, parse_file, mask_seq)
    logger.debug(f"Mask sequence after file pass: {mask_seq}")


    # Word masks: <word or phrase>
    pattern_word = r'<(?P<inner>[^<>]+?)>' # Non-greedy match inside angle brackets
    def parse_word(match_object):
        nonlocal inner_idx # Use inner_idx passed to function
        content = match_object.group('inner').strip()
        logger.debug(f"Parsing word mask: <{content}>")
        # Key depends on content, image, etc. Hash might be complex. Use unique ID for now.
        word_key = f"word_{inner_idx}_{len(val_masks)}_{content[:10]}" # Simple unique key
        try:
             # get_word_mask needs root object, image context
             # This is a major dependency on the calling environment structure
             if root is None: raise ValueError("'root' object needed for word masking is None.")
             mask_img = get_word_mask(root, frame_image, content) # Assume returns PIL 'L' or '1'
             if mask_img:
                 val_masks[word_key] = mask_img.convert('1') # Ensure binary
             else: # Handle word mask failure
                  logger.warning(f"Word mask <{content}> returned None. Using 'nowhere'.")
                  val_masks[word_key] = val_masks['nowhere']
        except Exception as e:
            logger.error(f"Error generating word mask <{content}>: {e}. Using 'nowhere'.")
            val_masks[word_key] = val_masks['nowhere']
        return f"{{{word_key}}}"
    mask_seq = re.sub(pattern_word, parse_word, mask_seq)
    logger.debug(f"Mask sequence after word pass: {mask_seq}")


    # --- Step 3: Boolean Operations ---
    # Order of operations matters: !, &, ^, \, | (Example: NOT, AND, XOR, DIFF, OR)

    # Helper to get mask from val_masks, ensures it's PIL '1'
    def get_mask_by_key(key):
        mask = val_masks.get(key)
        if mask is None:
             logger.error(f"Mask key '{key}' not found in val_masks!")
             return val_masks.get('nowhere', Image.new('1', (width, height), 0)) # Fallback
        if not isinstance(mask, Image.Image):
             logger.error(f"Value for key '{key}' is not a PIL Image!")
             return val_masks.get('nowhere', Image.new('1', (width, height), 0))
        if mask.mode != '1':
             return mask.convert('1') # Ensure binary mode
        return mask

    # Invert '!' operator: !{key}
    # Process repeatedly until no more inversions are found
    while True:
         pattern_invert = r'!\s*{?(?P<key>[^}\s]+)}?' # Match ! followed by optional {key} or just key
         match = re.search(pattern_invert, mask_seq)
         if not match: break

         key_to_invert = match.group('key')
         logger.debug(f"Processing invert: !{{{key_to_invert}}}")
         original_mask = get_mask_by_key(key_to_invert)
         inverted_key = f"inv_{key_to_invert}" # New key for inverted mask
         if inverted_key not in val_masks:
             val_masks[inverted_key] = ImageChops.invert(original_mask)

         # Replace the pattern with the new key
         mask_seq = mask_seq[:match.start()] + f"{{{inverted_key}}}" + mask_seq[match.end():]
         logger.debug(f"Mask sequence after invert pass: {mask_seq}")


    # Process binary operators in order (&, ^, \, |)
    operators = ['&', '^', '\\', '|'] # Define precedence / processing order
    op_functions = {
        '&': ImageChops.logical_and,
        '^': ImageChops.logical_xor,
        '\\': lambda m1, m2: ImageChops.logical_and(m1, ImageChops.invert(m2)), # Difference A \ B = A & !B
        '|': ImageChops.logical_or,
    }

    for op in operators:
        logger.debug(f"Processing operator: {op}")
        # Need robust regex to handle {key1} op {key2}
        # Escape backslash for regex pattern
        op_escaped = re.escape(op)
        pattern_op = r'{?(?P<key1>[^}\s]+)}?\s*' + op_escaped + r'\s*{?(?P<key2>[^}\s]+)}?'

        while True: # Process all occurrences of this operator
             match = re.search(pattern_op, mask_seq)
             if not match: break # No more occurrences of this operator

             key1 = match.group('key1')
             key2 = match.group('key2')
             logger.debug(f"Processing op {op}: {{{key1}}} {op} {{{key2}}}")

             mask1 = get_mask_by_key(key1)
             mask2 = get_mask_by_key(key2)

             result_key = f"res_{len(val_masks)}_{key1}_{op}_{key2}" # Generate unique key for result
             try:
                 result_mask = op_functions[op](mask1, mask2)
                 val_masks[result_key] = result_mask
             except Exception as e:
                 logger.error(f"Error applying operator {op} between {key1} and {key2}: {e}")
                 # Fallback to 'nowhere' or skip? For now, use 'nowhere' for safety.
                 val_masks[result_key] = val_masks['nowhere']


             # Replace the matched expression with the result key
             mask_seq = mask_seq[:match.start()] + f"{{{result_key}}}" + mask_seq[match.end():]
             logger.debug(f"Mask sequence after {op} pass: {mask_seq}")


    # --- Step 4: Output ---
    # Should be left with a single {final_key}
    final_pattern = r'^\s*{?(?P<final_key>[^}\s]+)}?\s*$'
    final_match = re.match(final_pattern, mask_seq)

    if final_match:
        final_key = final_match.group('final_key')
        logger.info(f"Mask composition finished. Final key: {final_key}")
        # Return the key itself, caller uses get_mask_by_key(key)
        return final_key
    else:
        logger.error(f"Mask composition did not result in a single key. Final sequence: {mask_seq}")
        # Fallback or raise error
        # Try to find any remaining key as a desperate measure?
        fallback_match = re.search(r'{?(?P<key>[^}\s]+)}?', mask_seq)
        if fallback_match:
             logger.warning("Falling back to first found key in sequence.")
             return fallback_match.group('key')
        else:
             logger.error("Could not find any valid key in final mask sequence.")
             return None # Indicate failure


def compose_mask_with_check(root, args, mask_seq, val_masks, frame_image):
    """ Wrapper for compose_mask that returns the final PIL mask ('L' mode) """
    width = getattr(args, 'width', 512)
    height = getattr(args, 'height', 512)

    # Ensure predefined masks exist and are PIL Images ('1' mode)
    for k, v in list(val_masks.items()): # Use list to allow modification during iteration if needed (though usually not)
         if not isinstance(v, Image.Image):
             logger.warning(f"Value for key '{k}' in initial val_masks is not PIL Image. Replacing with blank.")
             val_masks[k] = Image.new('1', (width, height), 0)
         elif v.mode != '1':
             val_masks[k] = v.convert('1') # Ensure binary

    if 'everywhere' not in val_masks: val_masks['everywhere'] = Image.new('1', (width, height), 1)
    if 'nowhere' not in val_masks: val_masks['nowhere'] = Image.new('1', (width, height), 0)

    try:
        final_key = compose_mask(root, args, mask_seq, val_masks, frame_image, 0)
        if final_key:
            final_mask_binary = val_masks.get(final_key)
            if final_mask_binary and isinstance(final_mask_binary, Image.Image):
                 # Convert final binary mask ('1') to grayscale ('L') for general use
                 final_mask_L = final_mask_binary.convert('L')
                 # Optional: Check the final mask
                 # checked_mask = check_mask_for_errors(final_mask_L, getattr(args, 'invert_mask', False))
                 # return checked_mask
                 return final_mask_L
            else:
                 logger.error(f"Final key '{final_key}' did not yield a valid PIL mask.")
                 return Image.new('L', (width, height), 0) # Return blank mask on failure
        else:
            logger.error("Mask composition failed to return a final key.")
            return Image.new('L', (width, height), 0) # Return blank mask on failure

    except ValueError as e: # Catch parsing errors like mismatched parenthesis
         logger.error(f"Error during mask composition (ValueError): {e}")
         return Image.new('L', (width, height), 0)
    except Exception as e: # Catch other unexpected errors
         logger.error(f"Unexpected error during mask composition: {e}")
         return Image.new('L', (width, height), 0)


def get_output_folder(output_path, batch_folder):
    """ Creates and returns the output folder path based on date and batch name. """
    # Use current date based on system time when function is called
    try:
         # Ensure base output path exists
         if not os.path.exists(output_path):
              os.makedirs(output_path, exist_ok=True)
              logger.info(f"Created base output directory: {output_path}")

         # Create date-based subfolder (YYYY-MM)
         date_folder = time.strftime('%Y-%m')
         date_path = os.path.join(output_path, date_folder)
         if not os.path.exists(date_path):
              os.makedirs(date_path, exist_ok=True)
              logger.info(f"Created date directory: {date_path}")

         # Create batch-specific subfolder if provided
         if batch_folder is not None and batch_folder.strip() != "":
              # Sanitize batch_folder name (optional, remove invalid chars)
              # safe_batch_folder = re.sub(r'[\\/*?:"<>|]', "_", batch_folder) # Basic sanitize
              final_path = os.path.join(date_path, batch_folder)
         else:
              final_path = date_path

         if not os.path.exists(final_path):
              os.makedirs(final_path, exist_ok=True)
              logger.info(f"Ensured final output directory exists: {final_path}")

         return final_path

    except Exception as e:
         logger.error(f"Error creating output folder structure: {e}")
         # Fallback to base output path if creation fails
         return output_path

# --- Image Saving Functions ---

# Thread target function for saving
def save_image_thread(image, path):
    """ Saves a PIL Image object to the specified path. """
    try:
        # Ensure directory exists right before saving
        dir_path = os.path.dirname(path)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path, exist_ok=True)

        image.save(path, "PNG") # Save as PNG by default
        logger.info(f"Saved image to {path}")
    except Exception as e:
        logger.error(f"Error saving image in thread to {path}: {e}")

# Main save_image function (using threading)
def save_image(image, image_type, filename, args, video_args, root, cls=None):
    """ Saves an image (PIL or CV2) to disk, potentially using threading. """
    # 'cls' argument seems unused here, removed from thread args unless needed later

    outdir = getattr(args, 'outdir', '.') # Get output directory from args
    store_in_ram = getattr(video_args, 'store_frames_in_ram', False)

    # Convert image to PIL if it's not already (needed for saving and RAM cache)
    img_pil = None
    if isinstance(image, Image.Image):
        img_pil = image.copy() # Work with a copy
    elif isinstance(image, np.ndarray):
        try:
            # Assume BGR if image_type hints at cv2, otherwise assume RGB
            if image_type == 'cv2': # Heuristic, might be wrong
                 img_pil = Image.fromarray(cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_BGR2RGB))
            else: # Assume RGB
                 img_pil = Image.fromarray(image.astype(np.uint8))
        except Exception as e:
             logger.error(f"Failed to convert NumPy image to PIL for saving: {e}")
             return # Don't save if conversion fails
    else:
        logger.error(f"Unsupported image type for saving: {type(image)}")
        return


    if store_in_ram:
        if hasattr(root, 'frames_cache') and isinstance(root.frames_cache, list):
            # Store PIL image in RAM cache
            logger.info(f"Storing frame in RAM cache: {filename}")
            root.frames_cache.append({
                'path': os.path.join(outdir, filename), # Store intended path
                'image': img_pil, # Store the PIL image itself
                'image_type': 'PIL' # Standardize cache type
            })
        else:
            logger.warning("store_frames_in_ram is True, but root.frames_cache is not available or not a list.")
            # Optionally save to disk as fallback? For now, just warn.
    else:
        # Save to disk using a separate thread
        full_path = os.path.join(outdir, filename)
        logger.info(f"Queueing image save to disk (threaded): {full_path}")

        # Create and start the thread
        try:
            thread = Thread(target=save_image_thread, args=(img_pil, full_path))
            thread.start()
            # Optionally store thread handles if you need to join them later
            # if not hasattr(root, 'save_threads'): root.save_threads = []
            # root.save_threads.append(thread)
        except Exception as e:
             logger.error(f"Failed to start save thread for {full_path}: {e}")
             # Fallback: save directly in main thread? (Could block)
             # try:
             #     logger.warning("Save thread failed, saving directly.")
             #     save_image_thread(img_pil, full_path)
             # except Exception as e2:
             #     logger.error(f"Direct save fallback also failed: {e2}")


def reset_frames_cache(root):
    if hasattr(root, 'frames_cache'):
        logger.info(f"Resetting frames cache (clearing {len(root.frames_cache)} items).")
        root.frames_cache = []
    gc.collect() # Trigger garbage collection


def dump_frames_cache(root):
    """ Saves all frames stored in root.frames_cache to disk. """
    if hasattr(root, 'frames_cache') and isinstance(root.frames_cache, list):
        num_frames = len(root.frames_cache)
        logger.info(f"Dumping {num_frames} frames from RAM cache to disk...")
        # Save sequentially for simplicity, threading might be complex here
        saved_count = 0
        for i, image_cache in enumerate(root.frames_cache):
            try:
                 path = image_cache.get('path')
                 image = image_cache.get('image') # Should be PIL image
                 if path and image and isinstance(image, Image.Image):
                     # Save using the thread function structure, but run directly here
                     save_image_thread(image, path) # This saves directly
                     saved_count += 1
                     # Optional: Log progress
                     if (i + 1) % 50 == 0 or (i + 1) == num_frames:
                          logger.info(f"Dumped {i+1}/{num_frames} frames...")
                 else:
                      logger.warning(f"Invalid cache entry at index {i}. Skipping dump.")
            except Exception as e:
                 logger.error(f"Error dumping frame {i} ({path}) from cache: {e}")

        logger.info(f"Finished dumping cache. Successfully saved {saved_count}/{num_frames} frames.")
        # Optionally clear cache after dumping?
        # reset_frames_cache(root)
    else:
         logger.info("No frame cache found or cache is not a list. Nothing to dump.")


# --- Flow/Remapping Functions (Originals) ---

def extend_flow(flow, target_w, target_h):
    """ Extends a smaller flow field to fit a larger target dimension by centering it. """
    flow_h, flow_w = flow.shape[:2]

    if flow_h >= target_h and flow_w >= target_w:
         # Flow is already larger or equal, just crop it
         logger.warning("extend_flow called with flow larger than target. Cropping.")
         return center_crop_image(flow, target_w, target_h)

    # Calculate offsets to center the original flow
    x_offset = max(0, (target_w - flow_w) // 2)
    y_offset = max(0, (target_h - flow_h) // 2)

    # Create base grid matching target dimensions (represents no flow initially)
    x_grid, y_grid = np.meshgrid(np.arange(target_w), np.arange(target_h))
    # new_flow = np.dstack((x_grid, y_grid)).astype(np.float32) # This is map, not flow!
    # Flow represents displacement, so extended area should have zero flow
    new_flow = np.zeros((target_h, target_w, 2), dtype=np.float32)


    # Adjust original flow vectors by the offset BEFORE placing them
    # flow_copy = flow.copy() # Important to modify a copy
    # flow_copy[:, :, 0] += x_offset # This is incorrect logic for flow vectors
    # flow_copy[:, :, 1] += y_offset # Flow vectors are relative displacements

    # Place the original flow vectors into the center of the new zero-flow field
    new_flow[y_offset:y_offset + flow_h, x_offset:x_offset + flow_w, :] = flow

    return new_flow


def remap(img, flow_map):
    """ Remaps image pixels based on a flow map using reflection padding. """
    # flow_map should be the absolute coordinates [map_x, map_y]
    h, w = img.shape[:2]

    # Check if flow_map dimensions match image dimensions
    map_h, map_w = flow_map.shape[:2]
    if h != map_h or w != map_w:
        logger.warning(f"Remap: Image shape {img.shape[:2]} and flow map shape {flow_map.shape[:2]} mismatch. Resizing map.")
        # Resize flow_map (absolute coordinates map)
        map_x = cv2.resize(flow_map[:,:,0], (w, h), interpolation=cv2.INTER_LINEAR)
        map_y = cv2.resize(flow_map[:,:,1], (w, h), interpolation=cv2.INTER_LINEAR)
        flow_map = np.dstack((map_x, map_y))


    # The original code used copyMakeBorder + extend_flow + remap + center_crop.
    # This seems overly complex if cv2.remap handles borders correctly.
    # Let's try direct remapping with border reflection.
    border_mode = cv2.BORDER_REFLECT_101

    # Ensure flow_map is float32
    if flow_map.dtype != np.float32:
        flow_map = flow_map.astype(np.float32)

    # Separate map_x and map_y for cv2.remap
    map_x, map_y = flow_map[:, :, 0], flow_map[:, :, 1]

    remapped_img = cv2.remap(img, map_x, map_y,
                           interpolation=cv2.INTER_LINEAR,
                           borderMode=border_mode)

    return remapped_img


# --- End of image_utils.py ---
