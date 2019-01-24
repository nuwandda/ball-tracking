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

fgbg = cv2.createBackgroundSubtractorMOG2()
# fff = cv2.bgsegm.createBackgroundSubtractorMOG()
# kernel = np.ones((7, 7), np.uint8)
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))

detect = 0
all_detect = 0

while True:
    # grab the current frame and initialize the occupied/unoccupied
    # text
    frame = vs.read()
    frame = frame if args.get("video", None) is None else frame[1]

    # if the frame could not be grabbed, then we have reached the end
    # of the video
    if frame is None:
        break

    # resize the frame, convert it to grayscale, and blur it
    frame = imutils.resize(frame, width=400)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # if the first frame is None, initialize it
    if firstFrame is None:
        firstFrame = gray
        continue

    # compute the absolute difference between the current frame and
    # first frame

    # absolute yerine bg sub kullan
    fgmask = fgbg.apply(gray)
    closing = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, kernel)
    opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel)
    # blurred = cv2.GaussianBlur(opening, (5, 5), 0)
    # frameDelta = cv2.absdiff(firstFrame, opening)
    # thresh = cv2.threshold(fgmask, 25, 255, cv2.THRESH_BINARY)[1]

    # dilate the thresholded image to fill in holes, then find contours
    # on thresholded image
    thresh = cv2.dilate(opening, kernel, iterations=2)
    # th = thresh[thresh < 240] = 0
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
        roi = frame[y: (y + h), x: (x + w)]
        key_points = detect_ball(roi)
        if len(key_points) > 0:
            detect = detect + 1
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        all_detect = all_detect + 1

    # show the frame and record if the user presses a key
    cv2.imshow("Frame", frame)
    cv2.imshow("Thresh", thresh)
    # cv2.imshow("Opening", opening)
    cv2.imshow("Mask", fgmask)
    key = cv2.waitKey(1) & 0xFF

    # if the `q` key is pressed, break from the lop
    if key == ord("q"):
        break

# cleanup the camera and close any open windows
vs.stop() if args.get("video", None) is None else vs.release()
print("Found ball in all frames: ", detect)
print("Detection in all frames: ", all_detect)
cv2.destroyAllWindows()
