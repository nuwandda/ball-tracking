import glob

from imutils.video import VideoStream
import argparse
import imutils
import time
import cv2
import numpy as np
import os
from frame import FrameObject
from detect_ball import DetectBall
from poly_fit import PolyFit
from io_operations import IOOperations
from perspective_transform import PerspectiveTransform as pt
from person_detection import PersonDetection as pd


# accuracy bulmak için train ve test dataları oluştur accuracy bul
# refactor et kodu


def on_mouse(event, real_x, real_y, flags, param):
    global src_points
    if event == cv2.EVENT_LBUTTONDOWN:
        cv2.circle(frame_clone, (real_x, real_y), 3, (0, 0, 255), thickness=-1)
        src_points.append([real_x, real_y])


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
# Flag for cropping image. True if ROI selected
backboard_found = False
five_frame_processed = 0
frame_buffer = []
object_counter = 0
play_count = 0
src_points = []
# destination points for homography estimation based on homo_dst.jpg
dst_img = cv2.imread("homo_dst.jpg")
dst_points = np.array([[239, 74],
                       [240, 290],
                       [375, 291],
                       [374, 74]])
dst_image_clone = dst_img.copy()

fgbg = cv2.createBackgroundSubtractorMOG2()
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
poly_constructor = PolyFit()
operations = IOOperations()

while True:
    # grab the current frame
    frame = vs.read()
    frame = frame if args.get("video", None) is None else frame[1]
    print("Please select with the order below:")
    print("Top Left, Bottom Left, Bottom Right and Top Right.")

    # resize the frame, convert it to grayscale
    frame = imutils.resize(frame, width=400)
    frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

    frame_clone = frame.copy()
    cv2.namedWindow("Homography")
    cv2.setMouseCallback("Homography", on_mouse)

    # keep looping until the 'e' key is pressed
    # this loop is for choosing real points for homography
    while True:
        # display the image and wait for a keypress
        cv2.imshow("Homography", frame_clone)
        key = cv2.waitKey(1) & 0xFF

        # if the 'r' key is pressed, reset the cropping region
        if key == ord("r"):
            src_points = []
            frame_clone = frame.copy()

        # if the 'e' key is pressed, break from the loop
        elif key == ord("e"):
            src_points = np.array(src_points)
            print("Source points for homography: ", src_points)
            cv2.destroyWindow("Homography")
            break
    break

while True:
    # grab the current frame
    frame = vs.read()
    frame = frame if args.get("video", None) is None else frame[1]

    # resize the frame, convert it to grayscale
    frame = imutils.resize(frame, width=400)
    frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    # Appends the current frame tho the buffer to use after
    frame_object = FrameObject(frame, object_counter)
    object_counter = object_counter + 1
    frame_buffer.append(frame_object)
    coef_x = []
    coef_y = []
    src_shot_location = np.array([])

    # Selects ROI if nothing selected
    if backboard_found is False:
        # Select ROI
        r = cv2.selectROI(frame)
        backboard_found = True

    # Crop image
    frame_copy = frame.copy()
    imCrop = frame_copy[int(r[1]):int(r[1] + r[3]), int(r[0]):int(r[0] + r[2])]

    # if the frame could not be grabbed, then we have reached the end
    # of the video
    if frame is None or imCrop is None:
        break

    # After finding movement inside the ROI, programs skips the 5 frames
    # to prevent from tracking same shot again
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
        points_of_ball = [int(x) + int(r[0]), int(y) + int(r[1]), w, h]
        roi = imCrop[y: (y + h), x: (x + w)]
        key_points = DetectBall.detect_ball(roi)
        if len(key_points) > 0:
            cv2.rectangle(imCrop, (x, y), (x + w, y + h), (0, 255, 0), 2)

            inner_count = 0
            if five_frame_processed == 0:
                search_x = points_of_ball[0]
                search_y = points_of_ball[1]
                search_w = points_of_ball[2]
                search_h = points_of_ball[3]
                path_base = 'play_' + str(play_count)
                path_search = 'play_' + str(play_count) + '/search_' + str(play_count)
                path_thresh = 'play_' + str(play_count) + '/search_thresh_' + str(play_count)
                path_coefs = 'play_' + str(play_count) + '/coefs' + str(play_count)
                os.makedirs(path_base)
                os.makedirs(path_search)
                os.makedirs(path_thresh)
                # Coefficients for the search areas in ball tracking
                point_coefficient = 0.8
                length_coefficient = 1.3
                # Points that pass as an argument to the model and these will limit the detection area
                yolo_x = 0
                yolo_y = 0
                yolo_w = 0
                yolo_h = 0

                for reverse_play in reversed(frame_buffer):
                    if inner_count == 35:
                        break

                    if inner_count > 31:
                        length_coefficient = 2

                    search_frame = reverse_play.getFrame()[
                                   int(search_y * point_coefficient): (int(search_y + search_h * length_coefficient)),
                                   int(search_x * point_coefficient): (int(search_x + search_w * length_coefficient))]

                    cv2.imwrite(os.path.join(path_search, 'search_' + str(inner_count) + '.jpg'), search_frame)

                    reverse_gray = cv2.cvtColor(reverse_play.getFrame(), cv2.COLOR_BGR2GRAY)
                    reverse_fgmask = fgbg.apply(reverse_gray)

                    reverse_closing = cv2.morphologyEx(reverse_fgmask, cv2.MORPH_CLOSE, kernel)
                    reverse_opening = cv2.morphologyEx(reverse_closing, cv2.MORPH_OPEN, kernel)

                    reverse_thresh = cv2.dilate(reverse_opening, kernel, iterations=2)

                    search_thresh = reverse_thresh[
                                    int(search_y * point_coefficient): (int(search_y + search_h * length_coefficient)),
                                    int(search_x * point_coefficient): (int(search_x + search_w * length_coefficient))]
                    cv2.imwrite(os.path.join(path_thresh, 'search_thresh_' + str(inner_count) + '.jpg'), search_thresh)

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
                        r_roi_temp = reverse_play.getFrame()[y + 10: (y + h), x + 10: (x + w)]
                        r_key_points = DetectBall.detect_ball(r_roi)
                        r_key_points_tmp = DetectBall.detect_ball(r_roi_temp)
                        if len(r_key_points) > 0:
                            if int(search_x * point_coefficient) < x < int(
                                    search_x + search_w * length_coefficient) and int(
                                    search_y * point_coefficient) < y < int(search_y + search_h * length_coefficient):
                                if inner_count != 0:
                                    if y + h > int(search_y + search_h * length_coefficient):
                                        # print("first", inner_count)
                                        cv2.rectangle(frame, (x, y), (x + w, y + h - search_h), (0, 255, 0), 2)
                                        yolo_x = x
                                        yolo_y = y
                                        yolo_w = w
                                        yolo_h = h
                                        if (x + w) / 2 != 0 and (y + h - search_h) / 2 != 0:
                                            if 24 < inner_count < 31:
                                                coef_x.append((x + w / 2))
                                                coef_y.append((y + h + 60))
                                            else:
                                                coef_x.append((x + w / 2))
                                                coef_y.append((y + h / 2 - search_h))
                                    else:
                                        # print("second", inner_count)
                                        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                                        if (x + w) / 2 != 0 and (y + h) / 2 != 0:
                                            coef_x.append((x + w / 2))
                                            coef_y.append((y + h / 2))

                                cv2.imwrite(os.path.join(path_base, "detected_" + str(play_count)) + ".jpg", frame)
                                if x == 0 or y == 0:
                                    continue
                                search_x = x
                                search_y = y
                                search_w = w
                                search_h = h
                    inner_count = inner_count + 1

                with open(path_coefs + '_x.txt', 'w') as f:
                    for index, item in enumerate(coef_x):
                        if index + 1 == len(coef_x):
                            f.write("%s" % item)
                        else:
                            f.write("%s," % item)

                with open(path_coefs + '_y.txt', 'w') as f:
                    for index, item in enumerate(coef_y):
                        if index + 1 == len(coef_y):
                            f.write("%s" % item)
                        else:
                            f.write("%s," % item)

                np_coefs_x = np.array(coef_x)
                np_coefs_y = np.array(coef_y)
                src_shot_location = pd.detect(frame, yolo_x, yolo_y, yolo_w, yolo_h)
                cv2.imwrite(os.path.join(path_base, "yolo_image" + str(play_count)) + ".jpg", frame)
                print("Shot location in real world: ", src_shot_location)

                # warped = warp.four_point_transform(frame, src_points)
                warped, h = pt.apply_homography(src_points, dst_points, frame, dst_img)
                cv2.imwrite(os.path.join(path_base, "warped_" + str(play_count)) + ".jpg", warped)

                t_shot_cart_coor, t_shot_homo_coor = pt.point_to_point_homography(src_shot_location, h)
                cv2.circle(dst_image_clone, (t_shot_cart_coor[0], t_shot_cart_coor[1]), 3, (0, 0, 255), thickness=-1)
                cv2.imwrite(os.path.join(path_base, "heatmap") + ".jpg", dst_image_clone)

                frame_buffer = []
                coef_x = []
                coef_y = []
                play_count = play_count + 1
            five_frame_processed = five_frame_processed + 1

    cv2.imshow("Cropped", imCrop)

    key = cv2.waitKey(1) & 0xFF

    # if the `q` key is pressed, break from the loop
    if key == ord("q"):
        break
