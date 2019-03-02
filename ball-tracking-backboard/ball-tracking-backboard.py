from imutils.video import VideoStream
import argparse
import imutils
import time
import cv2
import numpy as np


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
backboard = False

fgbg = cv2.createBackgroundSubtractorMOG2()
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))

while True:
    # grab the current frame
    frame = vs.read()
    frame = frame if args.get("video", None) is None else frame[1]

    # resize the frame, convert it to grayscale
    frame = imutils.resize(frame, width=400)

    if backboard is False:
        # Select ROI
        r = cv2.selectROI(frame)
        backboard = True
     
    # Crop image
    imCrop = frame[int(r[1]):int(r[1]+r[3]), int(r[0]):int(r[0]+r[2])]

    # if the frame could not be grabbed, then we have reached the end
    # of the video
    if frame is None or imCrop is None:
        break

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

    cv2.imshow("Masked", opening)
    cv2.imshow("Cropped", imCrop)

    key = cv2.waitKey(1) & 0xFF

    # if the `q` key is pressed, break from the lop
    if key == ord("q"):
        break
