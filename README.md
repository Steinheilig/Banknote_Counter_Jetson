# Banknote Counter - NVIDIA Jetson & LEGO Technic 
Using the NVIDIA Jetson Nano for banknote classification and to control LEGO power functions' motors and servomotors feeding single notes to realize a money counting LEGO MOC.

[<img src="./readme_images/MachineRun.JPG">](https://youtu.be/MnLnOKctatg)
Have a look at the [video](https://youtu.be/MnLnOKctatg) of the final working machine.<br>
Check also the [video](https://youtu.be/YSo6BacYehI) explaining the project in detail including motor control and training/inference with the NVIDIA Jetson.

## NVIDIA Jetson Nano Controlling LEGO Technic Power Functions' Motors and Servomotors
A total of 4 motors and a servomotor are controlled by the NVIDIA Jetson Nano. Therefore, a [PCA9685](https://learn.adafruit.com/16-channel-pwm-servo-driver?view=all) is connected to the Jetson Nano via I2C, offering 16 PWM (pulse-width modulation) channels. Two motor bridges (or H-bridges), more specifically two L293D chips, are used to drive the motors/servomotor with the 9V provided by the 9V Lego Power functions system and the PWM control channels of the PCA9685. <br>
A detailed video description of LEGO Power functions cable layout, PWM control of the motor/servomotor, etc. is given in the following:
- [LEGO Power Functions: Cable](https://youtu.be/tz1G_6vbO7Q)
- [LEGO Power Functions: Motor Control with PWM](https://youtu.be/gAPn_Jquxaw)
- [LEGO Power Functions: Servo Control with PWM](https://youtu.be/go4wQMvq7KU)
- [LEGO Power Functions: Control Motor/Servomotor via PCA9685](https://youtu.be/trT_Crm-CWE)
- [LEGO Power Functions: Employing the Jetson Nano to Control Motor/Servomotor via PCA9685](https://youtu.be/D2gSvXo0qT8)

## LEGO Technic Machine
In a nutshell, the servomotor is driving a first wheel using an overrunning or freewheel clutch. This first wheel pushes banknotes from a provides stack of notes. A first motor drives a second wheel which accelerates notes towards a third and final wheel. This wheel (driven by second motor) feeds single notes to a platform where a webcam (Logitec C270) is taking images from the note at the right time. A classification model infers the type of bank note based on this input image. After the note is classified a conveyor belt is used (third motor) to move the note out of the machine. Timing of the motor/servomotor control and the inference is done on the Jetson Nano. [Here's a video of the final machine in action.](https://youtu.be/MnLnOKctatg)

## Training a Banknote Classifier
The **Training.py** program is used for training the network. Either transfer learning (VGG16 base) or training a simple shallow CNN from scratch (5 layers) is implemented.
Six different classes (5, 10, 20, 50EUR, Background and Counterfeit Money) are defined and more than 500 images for each of the categories are taken und used for training. 
The training data set is recorded under various lighting conditions, different backgrounds, angles and distances using the [camera capture tool](https://github.com/dusty-nv/jetson-inference/blob/master/docs/pytorch-collect.md) provided by the NVIDIA AI hello world toolbox. <br>
- **Transfer Learning**<br>
The model is derived from a pretrained (ImageNet) VGG16 network, keeping the base model weights frozen.
Three dense layers (25,10,6) are stacked on top of the VGG16's final convolutional + maxpooling layers. 
A softmax layer is used after the last dense layer to estimate normalized confidences for each of the classes. <br>
For training all input images are converted to 60x60x3 and a batch size of 5 is chosen, to allow training on the Jetson Nano hardware. Furthermore, the graphical mode is deactivated (sudo init 3) to free up additional GPU/CPU memory. <br>
During training the data is augmented using slight random rotations, contrast variation, zoom, and translations. Adam optimizer with a sparse categorical crossentropy loss is used.<br> 
[Tensorflow 2.x with Keras](https://www.tensorflow.org/tutorials/images/transfer_learning) is used for model training with a [95/5% split](https://www.tensorflow.org/tutorials/load_data/images) of training and validation data set sizes. 
After around 10 epochs (roughly 1h per epoch) a training and validation accuracy of more than 80% is achieved. <br>
Running more epochs with larger batch size, increased input images size and more advanced (and deeper) pretrained networks, will most probably result in better performance but would require dedicated hardware for training. <br>
This [video](https://www.youtube.com/watch?v=YSo6BacYehI&t=323s) demonstrates the training data recording using the [camera capture tool](https://github.com/dusty-nv/jetson-inference/blob/master/docs/pytorch-collect.md) and the classifier training itself. 
- **Training Shallow CNN from Scratch**<br>
Alternatively, a shallow 5 (3 convolutional, 2 dense) layer network can be trained from scratch. Data preprocessing (rescaling & augmentation) as described before.
Training a network model from scratch given the small training set size turned out to be prone to overfitting on unrelevant features. Self-supervised pretraining (e.g. with convolutional autoencoder) might be an option to generate more robust feature extraction and finally a robust shallow CNN for this task. 

## Inference with the Banknote Classifier Model
The **Inference_MotorControl.py** code demonstrates inference with the trained model.
The trained Tensorflow model is loaded and used for inference on video frames provided by the webcam. Due to the fixed focus of the used webcam (Logitec C270) a compromise between banknote size and sharpness in the recorded images had to be found. As a result the camera was mounted on a hand crafted LEGO mount approx. 20cm above the conveyor belt. A better camera with variable focus and zoom will most probable increase the performance of such a setup.<br>
This [video](https://youtu.be/YSo6BacYehI?t=576) explains the inference part of the project. 
- **Performance Evaluation** <br>
A accuracy of more than 80% is achieved on training and validation set on the transfer learned model. The training set consists of 3632 and the validation set of 191 labeled images with roughly (+/- 2) balanced class distributions. <br>
The classifier is then tested with images recorded during running the banknote feeding machine. An additional video documentation of inference examples will be provided in the future. While high accurcies for the background, counterfeit money and 20EUR classes are achived. Accuracies for 5, 10, 50EUR classes are lower. Especially, the model has a hard time to distinguish notes from the 10EUR class from the other note classes. A confusion matrix is not provided to quantify this behavior.<br>
Note, that the model is trained on the (5W) Jetson Nano device only, limiting input image size, max batch sizes and network complexity to fit into CPU/GPU memory as well as the maximum number of epochs used for model training. 
- **Increase Performance** <br>
To achive better model performances a couple the following issues should be addressed
  -  better camera with variable focus and zoom 
  -  more training data (is always better...)
  -  network structure optimization (e.g. transfer learning with ResNet, using self-supervised pretraining,...)
  -  extensive training on dedicated hardware (allows e.g. larger batch sizes, more epochs)

## Banknote Counter - NVIDIA Jetson & LEGO Technic 
After initializing the PCA9685 library and loading the Tensorflow model, the following steps are performed by the Jetson Nano in a timed manner for each note to be classified
1) Starting motor #1, which will accelerate the note after it is pushed from the stack
2) Moving the servomotor left/right/, which will pushed notes from the stack 
3) Stopping motor #1, which will stop accelerate the note
4) Starting motor #2 only for a brief time span, which will feed a single note to the webcam
5) Image recorded from the webcam
6) Model inference of the banknote class based on the image 
7) Starting motor #3 to move the conveyor belt, which will transport the banknote out of the machine. <br>
**Inference_MotorControl.py** takes care of banknote class prediction with the loaded ANN model and controlling the motors/servomotors. 

## Literature
[1] [Adafruit PCA9685 16-Channel Servo Driver](https://learn.adafruit.com/16-channel-pwm-servo-driver?view=all)<br>
[2] [Paul McWhorter, AI on the Jetson Nano LESSON 31: Controlling Servos with the Jetson Nano using the PCA9685](https://youtu.be/8YKAtpPSEOk)<br>
[3] [Patrick M??ller (2016), Take Control Over Lego Power Functions: Learn how to control Lego motors and servos with your Arduino and build your own Android app to remote control your model.](https://create.arduino.cc/projecthub/Notthemarsian/take-control-over-lego-power-functions-ee0bfa)<br>
[4] [Dustin Franklin, NVIDIA Jetson - Hello AI World](https://github.com/dusty-nv/jetson-inference)<br>
[5] [Dustin Franklin, Camera Capture Tool](https://github.com/dusty-nv/jetson-inference/blob/master/docs/pytorch-collect.md)<br>
[6] [Tf2 Tutorial: Load and preprocess images](https://www.tensorflow.org/tutorials/load_data/images)<br>
[7] [Tf2 Tutorial: Transfer learning and fine-tuning](https://www.tensorflow.org/tutorials/images/transfer_learning) <br>
[8] [Robert Thas John, Transfer Learning with TF 2.0](https://towardsdatascience.com/transfer-learning-with-tf-2-0-ff960901046d)<br>
[9] [NVIDIA?? Jetson Nano???](https://www.nvidia.com/de-de/autonomous-machines/embedded-systems/jetson-nano/)<br>
[10] [Jetson Nano Developer Kit](https://developer.nvidia.com/embedded/jetson-nano-developer-kit)<br>
