import glob
import random
import time
import uuid
import image_fetch
import show_image

from PIL import Image


import numpy as np
import tensorflow as tf
import tensorflow_hub as hub

# Load model and set paths
model = hub.load('https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2')
generated_folder_path = ['./generated_images/*.jpg']
style_folder_path = ['./style_images/style_images_256/*.jpg']
style_image_path = ''

# Function for loading image from given path. Also converts it to right format.
def load_image(img_path):
    img = tf.io.read_file(img_path)
    img = tf.image.decode_image(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = img[tf.newaxis, :]
    return img

def get_random_img_path(path):
    all_images = glob.glob(random.choice(path))
    random_image_path = random.choice(all_images)
    return random_image_path

def tensor_to_image(tensor):
    tensor = tensor*255
    tensor = np.array(tensor, dtype=np.uint8)
    if np.ndim(tensor)>3:
        assert tensor.shape[0] == 1
        tensor = tensor[0]
    return Image.fromarray(tensor)


# Get first image from older generated images
# (only the first image of runtime is pre-generated)
first_image = get_random_img_path(generated_folder_path)
image_to_show = Image.open(first_image)

try:
    while True:
        # Start timer for calculating the required time to sleep
        tic = time.perf_counter()

        # Show the image saved in image_to_show
        root = show_image.showPIL(image_to_show)

        # After the image is presented, start generating the next image

        # Get random image and random style
        content_image_path = image_fetch.get_image()

        # Show older, already generated image if image fetching does not work
        if (content_image_path is None):
            impath = get_random_img_path(generated_folder_path)
            image_to_show = Image.open(impath)
            # Calculate how long the loop took, to calculate time to sleep. This is to have a new image presented
            # more accurately every X seconds. The calculation is needed because the processing time may vary,
            # which could create variance in time between two presented images.
            toc = time.perf_counter()
            if (float(toc - tic) < 10):
                time.sleep(10 - float(toc - tic))
            else:
                pass

            # Close the old image to prepare presenting next image
            show_image.destroyPIL(root)

        # If image fetch worked properly, proceed normally with processing
        else:

            style_image_path = get_random_img_path(style_folder_path)

            # For selecting style manually
            #style_image_path = './style_images_256\gogh (Custom).jpg'

            # Load content and style images
            content_image = load_image(content_image_path)
            #content_image = load_image('./content_images\img_324ae0c8-fa56-44b3-acc6-48ff800d84a9.jpg')
            #content_image = tf.image.resize(content_image, (180, 320))

            style_image = load_image(style_image_path)

            # Generate image using pretrained model
            stylized_image = model(tf.constant(content_image), tf.constant(style_image))[0]

            # Save generated image to a folder
            imname = 'img_' + str(uuid.uuid4())
            impath = './generated_images/' + imname + '.jpg'
            image = tensor_to_image(stylized_image[0])
            image.save(impath, format='PNG')

            image_to_show = Image.open(impath)

            # Calculate how long the loop took, to calculate time to sleep. This is to have a new image presented
            # more accurately every X seconds. The calculation is needed because the processing time may vary,
            # which could create variance in time between two presented images.
            toc = time.perf_counter()
            if (float(toc - tic) < 10):
                time.sleep(10 - float(toc - tic))
            else:
                pass

            # Close the old image to prepare presenting next image
            show_image.destroyPIL(root)



except KeyboardInterrupt:
    pass



# TODO
# Vanhojen kuvien poisto?