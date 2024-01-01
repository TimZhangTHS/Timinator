#imports
import numpy as np
import cv2
import pandas as pd
import mediapipe as mp
import math
import serial.tools.list_ports
import time




ports = serial.tools.list_ports.comports()
serialInst = serial.Serial('COM4',115200)
serialInst.setDTR(False)
serialInst.flushInput()
serialInst.setDTR(True)
time.sleep(6)

#portsList = []

#for onePort in ports:
#  portsList.append(str(onePort))
#  print(str(onePort))

#val = input("select Port: COM")

#for x in range(0,len(portsList)):
#  if portsList[x].startswith("COM"+str(val)):
#    portVar = "COM" + str(val)
#    print(portVar)

#serialInst.baudrate = 115200
#serialInst.port = portVar





#Media pipe set up
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

def draw_landmarks(image, holistic_results, draw_pose = True, draw_face = True, draw_hands = True):
  if(draw_face):
    mp_drawing.draw_landmarks(
      image,
      holistic_results.face_landmarks,
      mp_holistic.FACEMESH_CONTOURS
    )

    if(draw_pose):
      mp_drawing.draw_landmarks(
        image, 
        holistic_results.pose_landmarks,
        mp_holistic.POSE_CONNECTIONS
      )
    
    if(draw_hands):
      mp_drawing.draw_landmarks(
        image,
        holistic_results.left_hand_landmarks,
        mp_holistic.HAND_CONNECTIONS
      )

      mp_drawing.draw_landmarks(
        image,
        holistic_results.right_hand_landmarks,
        mp_holistic.HAND_CONNECTIONS
      )



def magnitude(vector):
  x = vector[0] * vector[0]
  y = vector[1] * vector[1]
  z =  vector[2] * vector[2]

  mag = math.sqrt(x+y+z)

  return mag

def normalize(vector):
  mag = magnitude(vector)

  x = vector[0] / mag
  y = vector[1] / mag
  z = vector[2] / mag

  return (x,y,z)

def dotProduct(vectorA, vectorB):
  vectorA_normal = normalize(vectorA)
  vectorB_normal = normalize(vectorB)

  x = vectorA_normal[0] * vectorB_normal[0]
  y = vectorA_normal[1] * vectorB_normal[1]
  z = vectorA_normal[2] * vectorB_normal[2]

  result = x + y + z

  return result


def angleBetweenVectors(vectorA, vectorB):
  dot = dotProduct(vectorA, vectorB)
  angle = math.acos(dot)
  return angle


def radainsToDegree(rad):
  return rad / math.pi * 180


def pointsToVector(pointA, pointB):
  x = pointA[0] - pointB[0]
  y = pointA[1] - pointB[1]
  z = pointA[2] - pointB[2]

  return (x,y,z)

def leftArmAngle(pose_landmarks):
  

  landmarks_list = pose_landmarks.landmark

  left_shoulder_landmark = (landmarks_list[11])
  left_elbow_landmark = (landmarks_list[13])
  left_hand_landmark = landmarks_list[15]

  a = (left_shoulder_landmark.x, left_shoulder_landmark.y, left_shoulder_landmark.z)
  b = (left_elbow_landmark.x, left_elbow_landmark.y, left_elbow_landmark.z)
  c = (left_hand_landmark.x, left_hand_landmark.y, left_hand_landmark.z)

  AB = pointsToVector(a,b)
  CB = pointsToVector(c,b)
  angle_ABC = radainsToDegree(angleBetweenVectors(AB, CB))/2

  angle_ABC = round(angle_ABC)



  

  return angle_ABC

def RightArmAngle(pose_landmarks):

  landmarks_list = pose_landmarks.landmark

  right_shoulder_landmark = (landmarks_list[12])
  right_elbow_landmark = (landmarks_list[14])
  right_hand_landmark = landmarks_list[16]

  a = (right_shoulder_landmark.x, right_shoulder_landmark.y, right_shoulder_landmark.z)
  b = (right_elbow_landmark.x, right_elbow_landmark.y, right_elbow_landmark.z)
  c = (right_hand_landmark.x, right_hand_landmark.y, right_hand_landmark.z)

  AB = pointsToVector(a,b)
  CB = pointsToVector(c,b)
  angle_ABC = radainsToDegree(angleBetweenVectors(AB, CB))
  print('Right arm: ', angle_ABC)

  return None

def main():
  average = []
  # Set up camera feed
  #capture = cv2.VideoCapture(0)  
  capture = cv2.VideoCapture(0)

  with mp_holistic.Holistic(
      min_detection_confidence = 0.5,
      min_tracking_confidence = 0.5
    ) as holistic:

    # Run while the camera is running
    while capture.isOpened():

      # Get the current frame from the camera
      success, frame = capture.read()

      # Check if we get a successful image
      if not success:
        print('Ignoring empty camera frame')
        continue
      # To improve performace, optionally mark the image as not writeable to
      # pass by reference.
      frame.flags.writeable = False

      # Convert the color to RGB for media pipe
      frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

      # Have MediaPipe process the image the Holestic solution
      results = holistic.process(frame)
      if results.pose_landmarks:
        angle_ABC = leftArmAngle(results.pose_landmarks)
        average.append(angle_ABC)
        if len(average) > 35:
          b = str(round(sum(average) / len(average)))
          serialInst.write(b.encode('utf-8'))
          print("average is: ", b)
          average = []
        #RightArmAngle(results.pose_landmarks)
      # Draw landmarks annotation on the image.
      frame.flags.writeable = True
      frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
      draw_landmarks(frame, results)

      #Show a live feed of results
      cv2.imshow('Camera Feed', cv2.flip(frame, 1))
      if cv2.waitKey(5) & 0xFF == 27:
        break



    
if __name__ == '__main__':
    main()





'''
https://developers.google.com/mediapipe/solutions/vision/hand_landmarker
https://developers.google.com/mediapipe/solutions/vision/pose_landmarker/
'''
