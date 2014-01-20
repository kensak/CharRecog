from PIL import Image
import os
from struct import *

def decompose(image_file_name, label_file_name, out_dir):
    
    if not os.path.isdir(out_dir):
        os.mkdir(out_dir)

    image_file = open(image_file_name, 'rb')
    label_file = open(label_file_name, 'rb')

    image_file.read(8) # skip
    tmpStr = image_file.read(8)
    rows, cols = unpack(">2i", tmpStr)

    label_file.read(4) # skip
    tmpStr = label_file.read(4)
    numImages = unpack(">i", tmpStr)[0]

    for i in range(numImages):
        data = image_file.read(rows * cols)
        image = Image.new("L", (cols, rows))
        label = ord(label_file.read(1))
        image.putdata(data)
        image.save(out_dir + "/" + str(label) + "_" + ("%05d" % i) + ".png", "PNG")

    image_file.close()
    label_file.close()

decompose("t10k-images.idx3-ubyte", "t10k-labels.idx1-ubyte", "MNIST_test_data")
decompose("train-images.idx3-ubyte", "train-labels.idx1-ubyte", "MNIST_train_data")
