# Handwritten Circuit Diagram to LTspice 

Object Detection on component on Handwritten Circuit Diagram, and convert them into LTspice simulation file.
![alt text](https://github.com/BIueMan/DL_Project-046211/blob/master/Images/fig71.png)
### Base on the YOLOv4 moudel

In this notebook, convert a Handwritten Circuit Diagram into LTspice file.
using deap learning and image prosesing, we can extract all the features we need in order to build a simulation file.

the algoritem can be divided into 3 part
1. Deep Learning on a YOLOv4 Base moudel, to extract the components and junctions locations
2. Image prossing, to extract all the wire and conation of the components
3. Convert into LTspice file base on all the features we have extracted

### part 1. deep learning model
In this part, we use a labeled Handwritten Circuit Diagram dataset[1]. we clean it with component we dont have alot of, and end up with 12 labels.
  capacitor, coil, resistor
  Power and current source
  diode and transistors
  Intersections and overleap on wires (Two different types of each)
  and finaly, text
  
![alt text](https://github.com/BIueMan/DL_Project-046211/blob/master/Images/YOLOv4%20model.png)

we Fine Tuning a YOLOv4 base model, that was train on general object detection. and got good resoults with a good yolo_loss:

we used in order to train the model:
  Epochs:          70
  Batch size:      2
  Subdivisions:    1
  Learning rate:   0.0001
  Optimizer:       adam
  Loss function:   yolo_loss
  Dataset classes: 12
        
![alt text](https://github.com/BIueMan/DL_Project-046211/blob/master/Images/Figure_1%2B2.png)

### part 2. image processing
to extract the wire and conetion between the compunents. we use a basic image processing technics, incloding:
1. convert the image into binary image
2. Erode and Dilate to remove noise
3. Remove label components from the image
4. Find Contours
5. Find connection using the contours

after we remove the labeled components from the images, we should stay only with the wire that conect the components. we can easy find them with "Find Contours", and use them to conect 2 compunent with the same contour.

![alt text](https://github.com/BIueMan/DL_Project-046211/blob/master/Images/image_proccesing.png)

### part 3. convort into LTspice file
finaly with all the Features, we can code circet using the location, and the conection of the components.

![alt text](https://github.com/BIueMan/DL_Project-046211/blob/master/Images/ltspice_circet.png)

We train the model on google colab, and so the can be easly rerun. how to run it is realdy wriden in the ipynd file.
see the Test file to test the project.
# Code
* training file - TRAIN-Circet_Object_Detector_YOLOv4.ipynb
* testing file, with the best weights - TEST_with_image-Circet_Object_Detector_YOLOv4.ipynb
* code to clean the dataset - basic_work_with_dataset.py
* the model - models.py
* the code to train the model - train.py

## in dir LTspice_tranforme
* the code that create component in an LTspice file - component_pleaser.py
* image proccesing code, that also create the LTspice file - build_circet.py (get as an input the labels of the image from the model, and the image)




# Reference
[1] Hand drawing circet diagram dataset - https://arxiv.org/abs/2107.10373 
[2] the model we toke and fine tuning -	https://github.com/roboflow-ai/pytorch-YOLOv4
[3] Yolov4 model article- https://towardsdatascience.com/yolo-v4-optimal-speed-accuracy-for-object-detection-79896ed47b50#6ac9 
