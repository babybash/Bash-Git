# save this as lab12.py
from flask import Flask

app = Flask(__name__)
@app.route("/")
def hello():

 from tensorflow.keras.applications import vgg16, inception_v3, resnet50, mobilenet

 # Load the VGG model
 vgg_model = vgg16.VGG16(weights='imagenet')

 # Load the Inception_V3 model
 inception_model = inception_v3.InceptionV3(weights='imagenet')

 # Load the ResNet50 model
 resnet_model = resnet50.ResNet50(weights='imagenet')

 # Load the MobileNet model
 mobilenet_model = mobilenet.MobileNet(weights='imagenet')

 # Commented out IPython magic to ensure Python compatibility.
 # %matplotlib inline
 import numpy as np

 from tensorflow.keras.preprocessing.image import load_img
 from tensorflow.keras.preprocessing.image import img_to_array

 import matplotlib.pyplot as plt

 def prepare_image(filename, size):
     # load an image in PIL format
     original = load_img(filename, target_size=size)

     # IN PIL - image is in (width, height, channel)
     print('PIL image size',original.size)
     plt.imshow(original)
     plt.show()

     # convert the PIL image to a numpy array
     # In Numpy - image is in (height, width, channel)
     numpy_image = img_to_array(original)
     # show the image
     # plt.imshow(np.uint8(numpy_image))
     # plt.show()
     print('numpy array size',numpy_image.shape)

     # Convert the image / images into batch format 
     # in the form (batchsize, height, width, channels) 
     image_batch = np.expand_dims(numpy_image, axis=0)
     print('image batch size', image_batch.shape)
     # show the image
     # plt.imshow(np.uint8(image_batch[0]))
     # plt.show()

     return image_batch

 image_batch = prepare_image('TestImages/banana.jpg', (224, 224))

 """# New Section"""

 from tensorflow.keras.applications.imagenet_utils import decode_predictions

 def predict(network, constructor, image_batch):
     model = constructor(weights='imagenet')
     processed_image = network.preprocess_input(image_batch.copy())

     # get the predicted probabilities for each class
     predictions = model.predict(processed_image)

     # get top 5 predictions which is the default
     label = decode_predictions(predictions)

     return label

 image_batch = prepare_image('TestImages/banana.jpg', (224, 224))

 print("---vgg16---")
 print(predict(vgg16, vgg16.VGG16, image_batch))
 print()

 print("---resnet50---")
 print(predict(resnet50, resnet50.ResNet50, image_batch))
 print()

 print("---mobilenet---")
 print(predict(mobilenet, mobilenet.MobileNet, image_batch))
 print()

 # inception_v3 requires image to be 299x299
 image_batch = prepare_image('TestImages/banana.jpg', (299, 299))

 print("---inception_v3---")
 print(predict(inception_v3, inception_v3.InceptionV3, image_batch))

