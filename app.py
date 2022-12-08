#!/usr/bin/env python
# coding: utf-8

# In[3]:


from flask import Flask, render_template, Response
import cv2
import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
import numpy as np


app=Flask(__name__)
def calculate_angle(a,b,c):
                        a = np.array(a) # First
                        b = np.array(b) # Mid
                        c = np.array(c) # End

                        radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
                        angle = np.abs(radians*180.0/np.pi)

                        if angle >180.0:
                            angle = 360-angle

                        return angle
cap = cv2.VideoCapture(1)


def gen_frames():   
            l_counter = 0
            r_counter = 0
            l_stage = None
            r_stage = None
## Setup mediapipe instance
            with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
                while cap.isOpened():
                    ret, frame = cap.read()
        
        # Recolor image to RGB
                    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    image.flags.writeable = False
      
        # Make detection
                    results = pose.process(image)

                    # Recolor back to BGR
                    image.flags.writeable = True
                    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        # Extract landmarks
                    try:
                        landmarks = results.pose_landmarks.landmark

                        # Get coordinates
                        l_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                                      landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                        l_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                                   landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                        l_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                                   landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]

                        r_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                                      landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
                        r_elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
                                   landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
                        r_wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
                                   landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]

                        # Calculate angle


                        l_angle = calculate_angle(l_shoulder, l_elbow, l_wrist)


                            # Visualize angle
                        cv2.putText(image, str(l_angle), 
                                           tuple(np.multiply(l_elbow, [640, 480]).astype(int)), 
                                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)

                            # Curl counter logic
                        if l_angle > 160:
                                l_stage = "down"
                        if l_angle < 20 and l_stage =='down':
                                l_stage="up"
                                l_counter +=1

                        r_angle = calculate_angle(r_shoulder, r_elbow, r_wrist)


                            # Visualize angle
                        cv2.putText(image, str(r_angle), 
                                           tuple(np.multiply(r_elbow, [640, 480]).astype(int)), 
                                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                                                )

                            # Curl counter logic
                        if r_angle > 160:
                                r_stage = "down"
                        if r_angle < 20 and r_stage =='down':
                                r_stage="up"
                                r_counter +=1


                    except:
                        pass

                    cv2.rectangle(image,(0,0),(225,70),(0,0,0),-1)
                    cv2.putText(image,"REPS",(15,12),
                                   cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),1,cv2.LINE_AA)
                    cv2.putText(image,str(l_counter),(10,60),
                                   cv2.FONT_HERSHEY_SIMPLEX,1.5,(255,255,255),2,cv2.LINE_AA)

                    cv2.putText(image,"STAGE",(75,12),
                                   cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),1,cv2.LINE_AA)
                    cv2.putText(image,l_stage,(70,60),
                                   cv2.FONT_HERSHEY_SIMPLEX,1.5,(255,255,255),2,cv2.LINE_AA)

                    cv2.rectangle(image,(640,0),(425,73),(0,0,0),-1)
                    cv2.putText(image,"REPS",(440,12),
                                   cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),1,cv2.LINE_AA)
                    cv2.putText(image,str(r_counter),(440,60),
                                   cv2.FONT_HERSHEY_SIMPLEX,1.5,(255,255,255),2,cv2.LINE_AA)

                    cv2.putText(image,"STAGE",(500,12),
                                   cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),1,cv2.LINE_AA)
                    cv2.putText(image,r_stage,(500,60),
                                   cv2.FONT_HERSHEY_SIMPLEX,1.5,(255,255,255),2,cv2.LINE_AA)



                    # Render detections
                    mp_drawing.draw_landmarks(image,results.pose_landmarks,mp_pose.POSE_CONNECTIONS,
                                             mp_drawing.DrawingSpec(color=(0,0,250),thickness=2,circle_radius=2),
                                             mp_drawing.DrawingSpec(color=(245,0,0),thickness=2,circle_radius=2))              
        
                    success, buffer = cv2.imencode('.jpg', image)
                    image = buffer.tobytes()
                    yield (b'--image\r\n'
                                       b'Content-Type: image/jpeg\r\n\r\n' + image + b'\r\n')


@app.route('/')
def index():
    return render_template('index.html')
@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=image')
if __name__=='__main__':
    app.run(debug=True)


# In[ ]:




