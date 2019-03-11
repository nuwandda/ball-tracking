from imutils.video import VideoStream
import argparse
import imutils
import time
import cv2
import numpy as np


class FrameObject:
    #An object to hold frame's index and actual frame
    def __init__(self,frame = None,index = None):
        self.frame = frame
        self.index = index

    def getFrame(self):
        return self.frame

    def getIndex(self):
        return self.index


def detect_ball(image):
    params = cv2.SimpleBlobDetector_Params()
    # Filter by Convexity
    params.filterByConvexity = True
    params.minConvexity = 0.87

    params.minThreshold = 10
    params.maxThreshold = 200

    # Filter by Inertia
    params.filterByInertia = True
    params.minInertiaRatio = 0.5

    # Filter by Circularity
    params.filterByCircularity = True
    params.minCircularity = 0.1

    params.filterByArea = True
    params.minArea = 200
    # params.maxArea = 400

    detector = cv2.SimpleBlobDetector_create(params)
    key_points = detector.detect(image)
    return key_points


ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", help="path to the video file")
ap.add_argument("-a", "--min-area", type=int, default=500, help="minimum area size")
args = vars(ap.parse_args())

if args.get("video", None) is None:
    vs = VideoStream(src=0).start()
    time.sleep(2.0)

else:
    vs = cv2.VideoCapture(args["video"])

firstFrame = None
backboard_found = False
five_frame_processed = 0
frame_buffer = []
object_counter = 0
play_count = 0

fgbg = cv2.createBackgroundSubtractorMOG2()
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))

while True:
    # grab the current frame
    frame = vs.read()
    frame = frame if args.get("video", None) is None else frame[1]

    # resize the frame, convert it to grayscale
    frame = imutils.resize(frame, width=400)
    frame_object = FrameObject(frame, object_counter)
    object_counter = object_counter + 1
    frame_buffer.append(frame_object)

    if backboard_found is False:
        # Select ROI
        r = cv2.selectROI(frame)
        backboard_found = True
     
    # Crop image
    frame_copy = frame.copy()
    imCrop = frame_copy[int(r[1]):int(r[1]+r[3]), int(r[0]):int(r[0]+r[2])]
    

    # if the frame could not be grabbed, then we have reached the end
    # of the video
    if frame is None or imCrop is None:
        break

    if five_frame_processed == 5:
        five_frame_processed = 0

    # resize the frame, convert it to grayscale
    gray = cv2.cvtColor(imCrop, cv2.COLOR_BGR2GRAY)

    # if the first frame is None, initialize it
    if firstFrame is None:
        firstFrame = gray
        continue

    fgmask = fgbg.apply(gray)
    closing = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, kernel)
    opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel)

    # dilate the opening image to fill in holes, then find contours
    # on thresholded image
    thresh = cv2.dilate(opening, kernel, iterations=2)
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if imutils.is_cv2() else cnts[1]

    # loop over the contours
    for c in cnts:
        # if the contour is too small, ignore it
        if cv2.contourArea(c) < args["min_area"]:
            continue

        # compute the bounding box for the contour, find ball with blob detection, draw it on the frame
        (x, y, w, h) = cv2.boundingRect(c)
        roi = imCrop[y: (y + h), x: (x + w)]
        key_points = detect_ball(roi)
        if len(key_points) > 0:
            cv2.rectangle(imCrop, (x, y), (x + w, y + h), (0, 255, 0), 2)

            inner_count = 0
            if five_frame_processed == 0:
                for reverse_play in reversed(frame_buffer):
                    if inner_count == 30:
                        break
                    # cv2.imwrite("reverse_frame_" + str(play_count) + ".jpg", reverse_play.getFrame())

                    reverse_gray = cv2.cvtColor(reverse_play.getFrame(), cv2.COLOR_BGR2GRAY)
                    reverse_fgmask = fgbg.apply(reverse_gray)
                    reverse_closing = cv2.morphologyEx(reverse_fgmask, cv2.MORPH_CLOSE, kernel)
                    reverse_opening = cv2.morphologyEx(reverse_closing, cv2.MORPH_OPEN, kernel)

                    reverse_thresh = cv2.dilate(reverse_opening, kernel, iterations=2)
                    reverse_cnts = cv2.findContours(reverse_thresh.copy(), cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE)
                    reverse_cnts = reverse_cnts[0] if imutils.is_cv2() else reverse_cnts[1]
                    
                    for r_c in reverse_cnts:
                        # if the contour is too small, ignore it
                        if cv2.contourArea(r_c) < args["min_area"]:
                            continue

                        # compute the bounding box for the contour, find ball with blob detection, draw it on the frame
                        (x, y, w, h) = cv2.boundingRect(r_c)
                        r_roi = reverse_play.getFrame()[y: (y + h), x: (x + w)]
                        r_key_points = detect_ball(r_roi)
                        if len(r_key_points) > 0:
                            cv2.rectangle(reverse_play.getFrame(), (x, y), (x + w, y + h), (0, 255, 0), 2)
                            cv2.imwrite("detected_" + str(play_count) + "_" + str(inner_count) + ".jpg", reverse_play.getFrame())
                            inner_count = inner_count + 1
                frame_buffer = []
                play_count = play_count + 1
            five_frame_processed = five_frame_processed + 1


    # cv2.imshow("Masked", opening)
    cv2.imshow("Cropped", imCrop)
    # print (frame_buffer)
    # print (frame_object.getFrame(), frame_object.getIndex())

    key = cv2.waitKey(1) & 0xFF

    # if the `q` key is pressed, break from the lop
    if key == ord("q"):
        break
