# NVIDIA Jetson Nano Banknote Counter Project
# Employing the NVIDIA Jetson Nano for banknote classification 
# and to control LEGO power functions' motors and servomotors 
# to feed single notes to realize a money counting LEGO MOC.
#
# INFERENCE AND MOTOR CONTROL
#
# 1) Controlling LEGO Power Functions motors and servomotors 
# - via PWM channels provided by PCA9685 
# - PCA9685 connected via I2C to the Jetson Nano
# - L293D (H-bridge)
# - LEGO Power Functions: 1 servomotor, 3 motors 
# - provide timed motor comands to feed single notes & allow banknote inference 
#
# 2) Banknote Classification using Tensorflow 2.x
# - load Tensorflow 2.x model
# - classify banknote
#
# https://www.youtube.com/channel/UCqL-arxKMK15cO7SVVBH2lA
# sh2021


import Adafruit_PCA9685 
import cv2 as cv 
import numpy as np 
import time
from imutils.video import FPS


Servo = [0,1]  # PWM channels on PCA9685 for each of the motors/servomotors
Motor1 = [4,5]
Motor2 = [8,9]
Motor3 = [12,13]
Classes = ['5EUR','10EUR','20EUR','50EUR','Background','COUNTERFEIT'] # class lables


# 1) Controlling LEGO Power Functions motors and servomotors 

servo = Adafruit_PCA9685.PCA9685(address=0x40,busnum=1)
print("init PCA9685 -done-")

def init_motors():
  # init all PWM duty cycle values to zeros
  for jj in range(2):
   servo.set_pwm(Servo[jj],0,0)
   servo.set_pwm(Motor1[jj],0,0)
   servo.set_pwm(Motor2[jj],0,0)
   servo.set_pwm(Motor3[jj],0,0)

def run_belt_step(dt=4):
  # run conveyor belt for dt seconds 
  servo.set_pwm(Motor3[0],0,4000)  # run motor3
  time.sleep(dt)
  servo.set_pwm(Motor3[0],0,0)  # stop motor3

def start_belt():
  # start conveyor belt
  servo.set_pwm(Motor3[0],0,4000)  # run motor3

def run_machine_step(run_belt=False):
  # run the banknote feeding machine
  # run_belt flag indicates whether conveyor belt should be used 
  servo.set_pwm(Motor1[1],0,4000)  # Start motor1 
  time.sleep(1)
  servo.set_pwm(Servo[1],0,4000)  #  Servo left
  time.sleep(0.5)
  servo.set_pwm(Servo[1],0,0)
  servo.set_pwm(Servo[0],0,4000)  #  Servo right
  time.sleep(0.5)
  servo.set_pwm(Servo[0],0,0)  # 
  time.sleep(0.5)
  servo.set_pwm(Motor2[0],0,3000)  # start motor2  
  time.sleep(0.5)
  servo.set_pwm(Motor2[0],0,0)  # stop motor2  
  servo.set_pwm(Motor1[1],0,0)  # stop motor1
  if run_belt:
    time.sleep(1)
    run_belt_step()

def demo_runs():
  # run the feeding mechanism 4 times 
  for zz in range(4):
    run_machine_step(run_belt=True)
    init_motors()
    time.sleep(2)

init_motors()  # initialize all PWM channels to zero


# 2) Banknote Classification using Tensorflow 2.x

import tensorflow as tf
from tensorflow.keras.models import load_model
print("init Tf -done-")

# load trained Tensorflow model from file 
width = height = 80 # input image size
dsize = (width, height)

model = load_model('model_vgg16_augmented_no_dropout_closer_bg_further_new_model7.h5') # load most recent model ;)
probability_model = tf.keras.Sequential([model,
                                         tf.keras.layers.Softmax()])   # extend the model with a softmax layer
print("model loaded -done-")

# start video capture / request low fps and low res.
cap = cv.VideoCapture("v4l2src device=/dev/video0 ! video/x-raw, format=YUY2, width=640, height=480, framerate=5/1 ! videoconvert ! video/x-raw, format=BGR ! appsink " , cv.CAP_GSTREAMER)
print("video capture -started-")
w = cap.get(cv.CAP_PROP_FRAME_WIDTH)
h = cap.get(cv.CAP_PROP_FRAME_HEIGHT)
fps = cap.get(cv.CAP_PROP_FPS)
print('Src opened, %dx%d @ %d fps' % (w, h, fps))
print('Wait for video sensor...')
time.sleep(1.0)  # not sure if it's necessary to wait for sensor...
fps = FPS().start()
fps.update()

cnt = 0 # count number of inference steps
use_softmax = True  # add softmax to normalize classification output 
debug_ = False      # debug mode flag
servo_on = False    # bool var. needed in debug mode
window_handle = cv.namedWindow('Test', cv.WINDOW_AUTOSIZE) # CV video
while cv.getWindowProperty('Test',0)>=0: # as long as the window exist
    cnt +=1

    for jj in range(7): # empty video buffer
      cap.grab()

    if cnt>60: # stop the machine after 60 inference steps
        break
    
    if cnt%2==0 and debug_: # if debug_ flag is set -> switch motor 1 on/off
        if servo_on == False:
            print('switch on motor 1 for testing...')
            servo.set_pwm(Motor1[0],0,4000)
        else:
            servo.set_pwm(Motor1[0],0,0)
        servo_on = not(servo_on)

    if cnt>11 and (cnt+4)%10==0: 
        # start the conveyor belt 4 inference steps before machine fed one note
        # print('start Belt at inference step',cnt)
        start_belt()

    return_value, image = cap.read() # read image from camera
    img = cv.resize(image, dsize)    # resize to appropriate input size
    img = (np.expand_dims(img,0))    # expend input dimensions for Tf
    
    if use_softmax:
       pred = probability_model.predict(img) # if we use softmax we get normalized probabilities
       pred = pred[0]
       #print(pred,'->',np.argmax(pred))
    else:
       pred = model.predict(img)
       pred = pred[0]
    
    Result = np.argmax(pred)  # prediciton: class of highest score/confidence 
    Confidence = pred[Result] # predicted value
    print(pred,'->',Classes[np.argmax(pred)])
    print(Classes[Result],'confidence:',Confidence)
    if Confidence > 0.20:  # chose low confidence value, due to low classification robustness 
      if Result==0:
                 cv.putText(image,"5 EUR",(100,440),cv.FONT_HERSHEY_SIMPLEX,2,(0,255,0),4,cv.LINE_AA)
      if Result==1:
                 cv.putText(image,"10 EUR",(100,440),cv.FONT_HERSHEY_SIMPLEX,2,(0,255,0),4,cv.LINE_AA)
      if Result==2:
                 cv.putText(image,"20 EUR",(100,440),cv.FONT_HERSHEY_SIMPLEX,2,(0,255,0),4,cv.LINE_AA)
      if Result==3:
                 cv.putText(image,"50 EUR",(100,440),cv.FONT_HERSHEY_SIMPLEX,2,(0,255,0),4,cv.LINE_AA)
      if Result==4:
                 cv.putText(image,"Background",(100,440),cv.FONT_HERSHEY_SIMPLEX,2,(255,255,255),4,cv.LINE_AA)
      if Result==5:
                 cv.putText(image,"COUNTERFEIT",(100,440),cv.FONT_HERSHEY_SIMPLEX,2,(0,0,255),4,cv.LINE_AA)
    cv.imshow('Test',image)
    # cv.imwrite('Vout'+str(cnt)+'.png',image)  # uncomment to save images to disc 
    fps.update()
    if cv.waitKey(1) & 0xFF==ord('q'):  # stop if keypress 'q'
        print('quit the window..')
        break

    if cnt>=10 and cnt%10==0:
      # Start banknote feeding machine at every 10th inference step
      run_machine_step(run_belt=False)
      print('Machine ran one step')

    if cnt>11 and (cnt+4-2)%10==0:
      # stop the conveyor belt 2 inference steps after it was started
      servo.set_pwm(Motor3[0],0,0)  # stop belt
      #print('stop belt')

fps.stop()
#print('approx FPS:',fps.fps()) # uncomment if you want to know the unoptimized truth 
del(cap) # stop video rec. 
cv.destroyAllWindows() # close video window 
init_motors()  # set all PWM channels to zero

