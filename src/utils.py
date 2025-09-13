from src.config import settings
from typing import List, Optional
import numpy as np
import cv2


def add_padding(x1, y1, x2, y2, parent_shape, padding_percent=0.1):
    """
    Adds padding to a bounding box and ensures the padded box stays within the parent image boundaries.

    Args:
        x1 (int): The x-coordinate of the top-left corner of the bounding box.
        y1 (int): The y-coordinate of the top-left corner of the bounding box.
        x2 (int): The x-coordinate of the bottom-right corner of the bounding box.
        y2 (int): The y-coordinate of the bottom-right corner of the bounding box.
        parent_shape (tuple): A tuple representing the shape of the parent image (height, width, channels).
        padding_percent (float, optional): The percentage of the bounding box's width/height to add as padding. Defaults to 0.1.

    Returns:
        tuple: A tuple containing the new (x1, y1, x2, y2) coordinates of the padded bounding box,
               clipped to the parent image dimensions.
    """
    # Get parent image dimensions
    parent_h, parent_w = parent_shape[:2]

    # Calculate current width and height
    width = x2 - x1
    height = y2 - y1

    # Calculate padding amounts
    pad_w = int(width * padding_percent)
    pad_h = int(height * padding_percent)

    # Calculate new coordinates with padding
    new_x1 = x1 - pad_w
    new_y1 = y1 - pad_h
    new_x2 = x2 + pad_w
    new_y2 = y2 + pad_h

    # Ensure new coordinates are within the parent image bounds
    final_x1 = max(0, new_x1)
    final_y1 = max(0, new_y1)
    final_x2 = min(parent_w, new_x2)
    final_y2 = min(parent_h, new_y2)

    return final_x1, final_y1, final_x2, final_y2


# def get_best_quality_crop(image_crops: List[np.ndarray]) -> Optional[np.ndarray]:
#     """
#     Selects the best quality image crop from a list based on a combined metric
#     of sharpness (Laplacian variance), contrast (standard deviation), and brightness (mean).
#     Crops are filtered based on a `quality_threshold` from settings.

#     Args:
#         image_crops (List[np.ndarray]): A list of numpy arrays, where each array
#                                         represents an image crop.

#     Returns:
#         Optional[np.ndarray]: The best quality image crop that meets the quality
#                               threshold, or None if no suitable crops are found
#                               or the input list is empty.
#     """
#     high_quality_crops = [
#         crop
#         for crop in image_crops
#         if crop is not None
#         and crop.size > 0
#         and (
#             (
#                 min(
#                     1.0,
#                     cv2.Laplacian(
#                         cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY), cv2.CV_64F
#                     ).var()
#                     / 100,
#                 )
#                 * 0.6
#                 + min(
#                     1.0,
#                     max(
#                         0.0,
#                         (cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY).std() - 20)
#                         / (128 - 20),
#                     ),
#                 )
#                 * 0.2
#                 + np.exp(
#                     -((cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY).mean() - 128) ** 2)
#                     / (2 * 128**2)
#                 )
#                 * 0.2
#             )
#             > settings.quality_threshold
#         )
#     ]
#     if not high_quality_crops:
#         return None

#     return max(high_quality_crops, key=lambda c: c.size)


def get_best_quality_crop(image_crops: List[np.ndarray]) -> Optional[np.ndarray]:
    """
    Selects the image crop with the highest sharpness score.

    This function iterates through a list of image crops, calculates the
    sharpness of each one using the variance of the Laplacian, and returns
    the crop with the maximum sharpness value.

    Args:
        image_crops (List[np.ndarray]): A list of numpy arrays, where each
                                       array represents an image crop.

    Returns:
        Optional[np.ndarray]: The sharpest image crop, or None if the input
                              list is empty or contains no valid crops.
    """
    best_crop = None
    max_sharpness = -1.0

    if not image_crops:
        return None

    for crop in image_crops:
        if crop is None or crop.size == 0:
            continue

        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

        sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
        if sharpness > max_sharpness:
            max_sharpness = sharpness
            best_crop = crop

    return best_crop


def get_best_quality_face_crop(image_crops: List[np.ndarray]) -> Optional[np.ndarray]:
    """
    Selects the best quality face crop from a list based on a combined score
    of sharpness (Laplacian variance), contrast (standard deviation), and brightness (mean).
    This function does not apply a strict threshold but returns the crop with the highest score.

    Args:
        image_crops (List[np.ndarray]): A list of numpy arrays, where each array
                                        represents a face image crop.

    Returns:
        Optional[np.ndarray]: The best quality face image crop, or None if the
                              input list is empty or contains only invalid crops.
    """
    scored_crops = []
    for crop in image_crops:
        if crop is None or crop.size == 0:
            continue

        gray_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

        laplacian_var = cv2.Laplacian(gray_crop, cv2.CV_64F).var()
        sharpness_score = min(1.0, laplacian_var / 100) * 0.6
        contrast_score = min(1.0, max(0.0, (gray_crop.std() - 20) / (128 - 20))) * 0.2
        brightness_score = np.exp(-((gray_crop.mean() - 128) ** 2) / (2 * 128**2)) * 0.2

        combined_score = sharpness_score + contrast_score + brightness_score

        scored_crops.append((combined_score, crop))

    if not scored_crops:
        return None

    best_score, best_crop = max(scored_crops, key=lambda item: item[0])
    return best_crop


def crop_image_by_percentage(
    image,
    start_row_percent,
    start_col_percent,
    end_row_percent,
    end_col_percent,
):
    """
    Crops an image based on percentage coordinates.

    Args:
        image (np.ndarray): The input image as a NumPy array.
        start_row_percent (float): The starting row as a percentage of the image height (0.0 to 1.0).
        start_col_percent (float): The starting column as a percentage of the image width (0.0 to 1.0).
        end_row_percent (float): The ending row as a percentage of the image height (0.0 to 1.0).
        end_col_percent (float): The ending column as a percentage of the image width (0.0 to 1.0).

    Returns:
        np.ndarray: The cropped image as a NumPy array.
        None: If an error occurs during cropping (e.g., FileNotFoundError, general Exception).

    Raises:
        Exception: Catches and prints any general exceptions that occur during processing.
    """

    try:

        height, width = image.shape[:2]

        start_row = int(height * start_row_percent)
        start_col = int(width * start_col_percent)

        # Calculate the ending pixel coordinates
        end_row = int(height * end_row_percent)
        end_col = int(width * end_col_percent)

        cropped_image = image[start_row:end_row, start_col:end_col]

        return cropped_image

    except FileNotFoundError as e:
        print(e)
        return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None


def resize_proportional(image, width=None, height=None, interpolation=cv2.INTER_AREA):
    """
    Resizes an image proportionally based on either a target width or height.

    Args:
        image (np.ndarray): The input image as a NumPy array.
        width (int, optional): The target width for the resized image. If provided,
                               height will be calculated to maintain aspect ratio.
                               Defaults to None.
        height (int, optional): The target height for the resized image. If provided,
                                width will be calculated to maintain aspect ratio.
                               Defaults to None.
        interpolation (int, optional): The interpolation method to use for resizing.
                                       Defaults to cv2.INTER_AREA.
    """
    (h, w) = image.shape[:2]

    if width is None and height is None:
        return image

    if width is None:
        ratio = height / float(h)
        new_dim = (int(w * ratio), height)
    else:
        ratio = width / float(w)
        new_dim = (width, int(h * ratio))

    resized = cv2.resize(image, new_dim, interpolation=interpolation)

    return resized
