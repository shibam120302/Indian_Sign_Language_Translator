# Indian-Sign-Language-Translator


This repository consists of the code utilized for creation of an Indian Sign Language Translator satisfying the following criteria :
- **Near-Real-Time Application**
- **Achieve background independence**
- **Attain Illumination independence**





We achieve these goals by providing the following features :
1. **Face Detection** 
: Used as an activation mechanism. We use Haar cascade models from the OpenCV library to detect faces in an image. When a face is detected, the program checks the next few consecutive frames, and when a threshold value of consecutive frames with a face in it is reached, the sign language system is triggered.

2. **Hand Detection** 
: The first step in preprocessing is preliminary hand detection method, which goes through every frame selected from the clip, and attempts to find a hand in them using a YOLO-v3 pre-trained network. 
If any hands are found in the frame, an extended bounding box is created around the hand(s). These images are then cropped to contain only the contents of the box, and are passed onto the next step of preprocessing which is resizing. If no hands are found, the frame is discarded entirely. 

3. **Skin Segmentation** 
: After cropping and resizing, the images are passed through a combination of HSV (Hue, Saturation, Value) and YCbCr (Luminance, Chrominance) based filters to segment out skin and remove background noise present in the box input.

4. **Sign Recognition** 
: The processed input is passed through a [SqueezeNet](https://arxiv.org/abs/1602.07360) model trained (via Transfer Learning) on a synthesized and cleaned Indian Sign Language dataset consisting of 10 classes, and ~2300+ images per class.



The work performed is divided into the following **folders** :

### Main App
The [App](https://github.com/shibam120302/Indian_Sign_Language_Translator) section consists of the files required to run the standalone webcam implementation of the translator. Contains :
- The trained model
- The hand segmentation network
- Preprocessing scripts
- Main application (main.py)


### Dataset Synthesis
Covers the [scripts](https://github.com/shibam120302/Indian_Sign_Language_Translator/tree/main/Dataset_Synthesis) used in :
- Creating new data, via modifications on brightness, clarity and picture quality (Synthesis)
- Cleaning noisy generated data from the previous step, by using the YOLO-v3 Hand Detection Network (Cleaning)

### Dataset Preprocessing
Contains the [scripts](https://github.com/shibam120302/Indian_Sign_Language-Translator/tree/main/Dataset_Preprocessing) used in order to perform pre-processing on the input dataset, including image upscaling, skin segmentation and hand centralization. These tasks are performed before entering the image dataset into the neural network.

### Model Training
Consists of the [notebook](https://github.com/shibam120302/Indian_Sign_Language_Translator/tree/main/Dataset_Preprocessing) used in order to train and save the SqueezeNet model used for the project. Originally made in Colab.

### Dependencies
- OpenCV
- Tensorflow
- Keras
- Numpy
- Pillow
- ImageAI

The specific versions are mentioned in requirements.txt

