# USAGE
# python allInOne.py --cascade haarcascade_frontalface_default.xml --shape-predictor shape_predictor_68_face_landmarks.dat
# python allInOne.py --cascade haarcascade_frontalface_default.xml --shape-predictor shape_predictor_68_face_landmarks.dat --alarm 1

# import the necessary packages
from imutils.video import VideoStream
from imutils.video import FileVideoStream
from imutils import face_utils
import numpy as np
import argparse
import imutils
import time
import dlib
import cv2

def euclidean_dist(ptA, ptB):
    # compute and return the euclidean distance between the two
    # points
    return np.linalg.norm(ptA - ptB)

def eye_aspect_ratio(eye):
    # compute the euclidean distances between the two sets of
    # vertical eye landmarks (x, y)-coordinates
    A = euclidean_dist(eye[1], eye[5])
    B = euclidean_dist(eye[2], eye[4])

    # compute the euclidean distance between the horizontal
    # eye landmark (x, y)-coordinates
    C = euclidean_dist(eye[0], eye[3])

    # compute the eye aspect ratio
    ear = (A + B) / (2.0 * C)

    # return the eye aspect ratio
    return ear
 
# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-c", "--cascade", required=False,
    help = "path to where the face cascade resides")
ap.add_argument("-p", "--shape-predictor", required=True,
    help= "path to facial landmark predictor")
ap.add_argument("-t", "--threshold",type = float, 
    help= "threshold for closed eye")
ap.add_argument("-f", "--outputFile", 
    help= "output file name")

args1 = ap.parse_args()
args = vars(args1)


 
# define two constants, one for the eye aspect ratio to indicate
# blink and then a second constant for the number of consecutive
# frames the eye must be below the threshold for to set off the
# alarm
EYE_AR_THRESH = 0.25
EYE_AR_CONSEC_FRAMES = 9

#persist for some time so that human eye can catch the signal
MAX_ALARM_PERSIST = 12

if args1.threshold:
    EYE_AR_THRESH = args["threshold"]
    print(EYE_AR_THRESH)

outFileBase = "earOut"



if args1.outputFile:
    outFileBase = args["outputFile"]

earOutFileName = outFileBase + ".dat"

noAlarmImgName = outFileBase + "_noalarm.png"
level1ImgName = outFileBase + "_level1.png"
level2ImgName = outFileBase + "_level2.png"
level4ImgName = outFileBase + "_level4.png"

isLevel4ImageSaved = False
isNoALARMImageSaved = False
isLevel1ImageSaved = False
isLevel2ImageSaved = False

# initialize the frame counter as well as a boolean used to
# indicate if the alarm is going off
COUNTER = 0
ALARM_ON = False

# load OpenCV's Haar cascade for face detection (which is faster than
# dlib's built-in HOG detector, but less accurate), then create the
# facial landmark predictor
print("[INFO] loading facial landmark predictor...")
if args1.cascade:
    detector = cv2.CascadeClassifier(args["cascade"])
    f_time = open("HaarTime.txt","w+")
else:
    detector = dlib.get_frontal_face_detector()
    f_time = open("HogTime.txt","w+")

f_blink = open("blink_timeStamp.txt","w+")

f_ear = open(earOutFileName,"w+")

predictor = dlib.shape_predictor(args["shape_predictor"])

# grab the indexes of the facial landmarks for the left and
# right eye, respectively
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

# start the video stream thread
print("[INFO] starting video stream thread...")
# vs = VideoStream(src=0).start()
#vs = VideoStream(usePiCamera=True).start()
vs = FileVideoStream("ridy.avi").start()
time.sleep(1.0)

#slide alg 4
blink_count = 0
time_close = 0
isClose = False

face_detect_count = 0
face_detect_time_acc =0
t_base = time.time()

# array of EARs

#algorithm 4 / slide alg 1
l_60=[1]

#algorithm 3 / slide alg 2
l_30 = []


alarm_on_persistent_count = 0


current_alarm_level = 0
new_alarm_level = 0



# loop over frames from the video stream
while True:
    # grab the frame from the threaded video file stream, resize
    # it, and convert it to grayscale
    # channels)
    
    t1=time.time()
    frame = vs.read()
    frame = imutils.resize(frame, width=450)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


    # detect faces in the grayscale frame
    if args1.cascade:
        rects = detector.detectMultiScale(gray, scaleFactor=1.1, 
            minNeighbors=5, minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE)
    else:
        rects = detector(gray, 1)

    t2=time.time()

    if args1.cascade:
        dlib_rects = []
        face_count = 0
        for (x,y,w,h) in rects:
            dlib_rects.append(dlib.rectangle((x), int(y), int(x+w),int(y+h)))
            face_count = face_count +1
    else:
        dlib_rects = rects
        face_count = len(dlib_rects)

    

    
    for rect in dlib_rects:
        # determine the facial landmarks for the face region, then
        # convert the facial landmark (x, y)-coordinates to a NumPy
        # array
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        # extract the left and right eye coordinates, then use the
        # coordinates to compute the eye aspect ratio for both eyes
        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]
        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)

        # average the eye aspect ratio together for both eyes
        ear = (leftEAR + rightEAR) / 2.0

            
        # compute the convex hull for the left and right eye, then
        # visualize each of the eyes
        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

        # check to see if the eye aspect ratio is below the blink
        # threshold, and if so, increment the blink frame counter
        if face_count == 1:

            new_alarm_level = 0
            
            #1
            l_60.append(ear)
            if len(l_60) > 60:
                del l_60[0]

            if len(l_60) == 60:    
                #sum the number of times the ear value falls under threshold, number of blinks
                number = sum(1 for i in l_60 if l_60.index(i)+1 < len(l_60) and i > EYE_AR_THRESH and l_60[l_60.index(i)+1] < EYE_AR_THRESH)
                if number > 4:
                    new_alarm_level = 1

            #2
            l_30.append(ear)
            if len(l_30) > 30:
                del l_30[0]

            avgear = sum(l_30)/len(l_30)
            if avgear < EYE_AR_THRESH:
                new_alarm_level = 2
                
            #3
            if len(l_30) == 30:   
                number = sum(1 for i in l_30 if i < EYE_AR_THRESH)
                if number > 15:
                    new_alarm_level = 3
              
            #4    
            if ear < EYE_AR_THRESH:
                COUNTER += 1

                # if the eyes were closed for a sufficient number of
                # frames, then sound the alarm
                if COUNTER >= EYE_AR_CONSEC_FRAMES:
                    current_alarm_level = 4
                    # if the alarm is not on, turn it on
                    #if not ALARM_ON:
                    #    ALARM_ON = True



                    # draw an alarm on the frame
                    #cv2.putText(frame, "DROWSINESS ALERT!", (10, 30),
                    #    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                if not isClose:
                    time_close = t1
                    isClose = True
                
            # otherwise, the eye aspect ratio is not below the blink
            # threshold, so reset the counter and alarm
            else:
                COUNTER = 0
                ALARM_ON = False
                if isClose:
                    isClose = False
                    blink_count = blink_count +1
                    f_blink.write("%f %f\n"%(t1-t_base,t1-time_close))

            if current_alarm_level < new_alarm_level:
                current_alarm_level = new_alarm_level
                alarm_on_persistent_count = 0
            else:
                if current_alarm_level > new_alarm_level:
                    if alarm_on_persistent_count >= MAX_ALARM_PERSIST:
                        current_alarm_level = new_alarm_level
                        alarm_on_persistent_count = 0
                    else:
                        alarm_on_persistent_count = alarm_on_persistent_count +1

                
            # draw an alarm on the frame

            if current_alarm_level > 0:
                cv2.putText(frame, "DROWSINESS ALERT %d "%(current_alarm_level), (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # draw the computed eye aspect ratio on the frame to help
            # with debugging and setting the correct eye aspect ratio
            # thresholds and frame counters
            cv2.putText(frame, "EAR: {:.3f}".format(ear), (300, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)


            if not isNoALARMImageSaved:
                if current_alarm_level <1:
                    isNoALARMImageSaved = True
                    cv2.imwrite(noAlarmImgName,frame)

            if not isLevel4ImageSaved:
                if current_alarm_level == 4:
                    isLevel4ImageSaved = True
                    cv2.imwrite(level4ImgName,frame)


            if not isLevel1ImageSaved:
                if current_alarm_level == 1:
                    isLevel1ImageSaved = True
                    cv2.imwrite(level1ImgName,frame)

            if not isLevel2ImageSaved:
                if current_alarm_level == 2:
                    isLevel2ImageSaved = True
                    cv2.imwrite(level2ImgName,frame)
                    
            f_ear.write("%f %f\n"%(t1-t_base,ear))
            
        else:
            #if more than one face in camera, then abort detection  
            #reset detection counters
            l_30.clear()
            l_60.clear()
            COUNTER = 0
            current_alarm_level = 0
            cv2.putText(frame, "%d faces"%(face_count), (300, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            f_ear.write("%f %f\n"%(t1-t_base,1))
 
    # show the frame
    t3 = time.time()

    if face_count < 1:
        l_30.clear()
        l_60.clear()
        current_alarm_level = 0
        cv2.putText(frame, "%d faces"%(face_count), (300, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        f_ear.write("%f %f\n"%(t1-t_base,1))
    
    # face detect cycle count
    face_detect_time_acc = face_detect_time_acc + (t2-t1)
    face_detect_count = face_detect_count +1

    if face_detect_count > 100:
        f_time.write("%f %f   %f\n"%(t2-t1,t3-t2,face_detect_time_acc/face_detect_count))
    else:
        f_time.write("%f %f\n"%(t2-t1,t3-t2))
        
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF
    # t2 = time.time()
    # print(t2-t1)
    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        f_time.close()
        f_blink.close()
        break

# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()
