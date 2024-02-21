import math
import cv2

import numpy as np

def load(image_path):
    """Loads an image from a file path.

    HINT: Look up `cv2.imread()` function - 
          whatch out  for the returned color format ! Check the following link for some fun : 
          https://www.learnopencv.com/why-does-opencv-use-bgr-color-format/

    Args:
        image_path: file path to the image.

    Returns:
        out: numpy array of shape(image_height, image_width, 3).
    """
    out = None

    ### VOTRE CODE ICI - DEBUT
    # Utilisez cv2.imread - le format RGB doit être retourné
    out = cv2.imread(image_path)
    if out is None:
        raise AssertionError("File doesn't exist!!!")
    out = cv2.cvtColor(out, cv2.COLOR_BGR2RGB)
    ### VOTRE CODE ICI - FIN
    # Let's convert the image to be between the correct range.
    out = out.astype(np.float32) / 255
    return out


def dim_image(image):
    """Change the value of every pixel by following

                        x_n = 0.5*x_p^2

    where x_n is the new value and x_p is the original value.

    Args:
        image: numpy array of shape(image_height, image_width, 3).

    Returns:
        out: numpy array of shape(image_height, image_width, 3).
    """

    out = None

    ### VOTRE CODE ICI - DEBUT
    out = 0.5 * (image ** 2)
    ### VOTRE CODE ICI - FIN

    return out


def convert_to_grey_scale(image):
    """Change image to gray scale.

    HINT: see if you can use  the opencv function `cv2.cvtColor()` 
    Args:
        image: numpy array of shape(image_height, image_width, 3).

    Returns:
        out: numpy array of shape(image_height, image_width).
    """
    out = None

    ### VOTRE CODE ICI - DEBUT
    out = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
    ### VOTRE CODE ICI - FIN

    return out


def rgb_exclusion(image, channel):
    """Return image **excluding** the rgb channel specified

    Args:
        image: numpy array of shape(image_height, image_width, 3).
        channel: str specifying the channel. Can be either "R", "G" or "B".

    Returns:
        out: numpy array of shape(image_height, image_width, 3).
    """

    out = None

    ### VOTRE CODE ICI - DEBUT
    out = np.copy(image)
    if channel == 'R':
        out[:, :, 0] = 0
    elif channel == 'G':
        out[:, :, 1] = 0
    elif channel == 'B':
        out[:, :, 2] = 0
    else:
        assert ValueError("Value doesn't exist in this space !!!")
    ### VOTRE CODE ICI - FIN
    return out


def lab_decomposition(image, channel):
    """Decomposes the image into LAB and only returns the channel specified.

    Args:
        image: numpy array of shape(image_height, image_width, 3).
        channel: str specifying the channel. Can be either "L", "A" or "B".

    Returns:
        out: numpy array of shape(image_height, image_width).
    """

    out = None

    ### VOTRE CODE ICI - DEBUT
    out = cv2.cvtColor(image,cv2.COLOR_RGB2LAB)
    if channel == 'L':
        return out[:, :, 0]
    elif channel == 'A':
        return out[:, :, 1]
    elif channel == 'B':
        return out[:, :, 2]
    else:
        assert ValueError("Value doesn't exist in this space !!!")
    ### VOTRE CODE ICI - FIN
    return out


def hsv_decomposition(image, channel='H'):
    """Decomposes the image into HSV and only returns the channel specified.

    Args:
        image: numpy array of shape(image_height, image_width, 3).
        channel: str specifying the channel. Can be either "H", "S" or "V".

    Returns:
        out: numpy array of shape(image_height, image_width).
    """
    out = None

    ### VOTRE CODE ICI - DEBUT
    out = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    if channel == 'H':
        return out[:, :, 0]
    elif channel == 'S':
        return out[:, :, 1]
    elif channel == 'V':
        return out[:, :, 2]
    else:
        assert ValueError("Value doesn't exist in this space !!!")
    ### VOTRE CODE ICI - FIN

    return out


def mix_images(image1, image2, channel1, channel2):
    """Combines image1 and image2 by taking the left half of image1
    and the right half of image2. The final combination also excludes
    channel1 from image1 and channel2 from image2 for each image.

    HINTS: Use `rgb_exclusion()` you implemented earlier as a helper
    function. Also look up `np.concatenate()` to help you combine images.

    Args:
        image1: numpy array of shape(image_height, image_width, 3).
        image2: numpy array of shape(image_height, image_width, 3).
        channel1: str specifying channel used for image1.
        channel2: str specifying channel used for image2.

    Returns:
        out: numpy array of shape(image_height, image_width, 3).
    """

    out = None
    ### VOTRE CODE ICI - DEBUT
    out = np.concatenate(
                    (
                            rgb_exclusion(image1, channel1)[:, :image1.shape[1] // 2, :],
                            rgb_exclusion(image2, channel2)[:, image2.shape[1] // 2:, :]
                            ),
                    axis=1)
    ### VOTRE CODE ICI - FIN

    return out

def mix_quadrants(image):
    """
    This function takes an image, and performs a different operation
    to each of the 4 quadrants of the image. Then it combines the 4
    quadrants back together.

    Here are the 4 operations you should perform on the 4 quadrants:
        Top left quadrant: Remove the 'R' channel using `rgb_exclusion()`.
        Top right quadrant: Dim the quadrant using `dim_image()`.
        Bottom left quadrant: Brighthen the quadrant using the function:
            x_n = x_p^0.5
        Bottom right quadrant: Remove the 'R' channel using `rgb_exclusion()`.

    Args:
        image1: numpy array of shape(image_height, image_width, 3).

    Returns:
        out: numpy array of shape(image_height, image_width, 3).
    """
    out = None

    ## VOTRE CODE ICI - DEBUT
    out = np.concatenate(
    (
        # Top
        np.concatenate(
            (
                (rgb_exclusion(image, "R")[:(image.shape[0] // 2), :(image.shape[1] // 2), :]), # Top left
                (image[(image.shape[0] // 2):, :(image.shape[1] // 2), :] ** 0.5) # Bottom left
            ),
            axis=0
        ),
        np.concatenate(
            (
                # Bottom
                (dim_image(image[:(image.shape[0] // 2), (image.shape[1] // 2):, :])), # Top right
                (rgb_exclusion(image, "R")[(image.shape[0] // 2):, (image.shape[1] // 2):, :]) # Bottom right
            ),
            axis=0
        )
    ),
    axis=1
    )
    ### VOTRE CODE ICI - FIN

    return out
