# Utilizes code from
# https://github.com/sankit1/cv-tricks.com/tree/master/Tensorflow-tutorials/tutorial-2-image-classifier
# in predicty.py to classify the most recent image in a folder. Used for our demo presentation.
import tensorflow as tf
import numpy as np
import os,glob,cv2
import sys,argparse

def predictImage(image_path):
    # First, pass the path of the image
    dir_path = os.path.dirname(os.path.realpath(__file__))

    # image_path=sys.argv[1]
    # image_path='testing_data/threev1/18threev1.jpeg'

    filename = dir_path +'/' +image_path
    image_size=128
    num_channels=3
    images = []
    # Reading the image using OpenCV
    image = cv2.imread(filename)
    # Resizing the image to our desired size and preprocessing will be done exactly as done during training
    image = cv2.resize(image, (image_size, image_size),0,0, cv2.INTER_LINEAR)
    images.append(image)
    images = np.array(images, dtype=np.uint8)
    images = images.astype('float32')
    images = np.multiply(images, 1.0/255.0)
    #The input to the network is of shape [None image_size image_size num_channels]. Hence we reshape.
    x_batch = images.reshape(1, image_size,image_size,num_channels)

    ###
    # with tf.device('/gpu:0'):
    ## Let us restore the saved model
    sess = tf.Session()
    # Step-1: Recreate the network graph. At this step only graph is created.
    saver = tf.train.import_meta_graph('trained_models/one-through-five-modelflp.meta')
    # Step-2: Now let's load the weights saved using the restore method.
    # saver.restore(sess, tf.train.latest_checkpoint('./trained_models/'))
    saver.restore(sess, 'trained_models/one-through-five-modelflp')

    # Accessing the default graph which we have restored
    graph = tf.get_default_graph()
    ###

    # Now, let's get hold of the op that we can be processed to get the output.
    # In the original network y_pred is the tensor that is the prediction of the network
    y_pred = graph.get_tensor_by_name("y_pred:0")

    ## Let's feed the images to the input placeholders
    x= graph.get_tensor_by_name("x:0")
    y_true = graph.get_tensor_by_name("y_true:0")
    y_test_images = np.zeros((1, len(os.listdir('training_data'))))


    ### Creating the feed_dict that is required to be fed to calculate y_pred
    feed_dict_testing = {x: x_batch, y_true: y_test_images}
    result=sess.run(y_pred, feed_dict=feed_dict_testing)
    # print(result)
    return result
    # result is of this format [probabiliy_of_rose probability_of_sunflower]

max_mtime = 0
startpath = 'testing_data/onev1/' # Needs a '/' at the end
for fname in os.listdir(startpath):
    full_path = startpath + fname
    mtime = os.stat(full_path).st_mtime
    if mtime > max_mtime:
        max_mtime = mtime
        max_dir = startpath
        max_file = full_path
result = predictImage(max_file).tolist()
maxIndex = result[0].index(max(result[0]))
handDict = {0:'five',1:'four',2:'one',3:'three',4:'two'}

correctPrediction = "correct" if handDict[maxIndex] in startpath else "incorrect"
import tkinter as tk
from PIL import ImageTk, Image
root = tk.Tk()
root.title("How many fingers are you holding up?")
root.geometry("800x400")
imagepath = max_file
img = ImageTk.PhotoImage(Image.open(imagepath))
panel = tk.Label(root, image = img)
panel.pack(side = "bottom", fill = "both", expand = "yes")
displaytext = tk.Label(root, height=5, width=30, text= (handDict[maxIndex] + " -- " + str(correctPrediction)))
displaytext.config(font=("Verdana", 32))
displaytext.pack(side="top")
root.mainloop()
