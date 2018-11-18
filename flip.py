from PIL import Image
import os
def flip_image(image_path, saved_location):
    """
    Flip or mirror the image

    @param image_path: The path to the image to edit
    @param saved_location: Path to save the cropped image
    """
    image_obj = Image.open(image_path)
    rotated_image = image_obj.transpose(Image.FLIP_LEFT_RIGHT)
    rotated_image.save(saved_location)

startpath = 'training_data/'
for directory in os.listdir(startpath):
    directorypath = startpath + directory + '/'
    for filename in os.listdir(directorypath):
        accesspath = directorypath + filename
        newpath = directorypath + 'flp' + filename
        print(accesspath)
        flip_image(accesspath, newpath)
