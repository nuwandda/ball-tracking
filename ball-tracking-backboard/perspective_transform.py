# import the necessary packages
import numpy as np
import cv2


class PerspectiveTransform:
    def __init__(self):
        pass

    @staticmethod
    def order_points(pts):
        # initialize a list of coordinates that will be ordered
        # such that the first entry in the list is the top-left,
        # the second entry is the top-right, the third is the
        # bottom-right, and the fourth is the bottom-left
        rect = np.zeros((4, 2), dtype="float32")

        # the top-left point will have the smallest sum, whereas
        # the bottom-right point will have the largest sum
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]

        # now, compute the difference between the points, the
        # top-right point will have the smallest difference,
        # whereas the bottom-left will have the largest difference
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]

        # return the ordered coordinates
        return rect

    @staticmethod
    def four_point_transform(image, pts):
        # obtain a consistent order of the points and unpack them
        # individually
        rect = pts
        (tl, tr, br, bl) = rect

        # compute the width of the new image, which will be the
        # maximum distance between bottom-right and bottom-left
        # x-coordinates or the top-right and top-left x-coordinates
        widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
        widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
        maxWidth = max(int(widthA), int(widthB))

        # compute the height of the new image, which will be the
        # maximum distance between the top-right and bottom-right
        # y-coordinates or the top-left and bottom-left y-coordinates
        heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
        heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
        maxHeight = max(int(heightA), int(heightB))

        # now that we have the dimensions of the new image, construct
        # the set of destination points to obtain a "birds eye view",
        # (i.e. top-down view) of the image, again specifying points
        # in the top-left, top-right, bottom-right, and bottom-left
        # order
        dst = np.array([
            [0, 0],
            [maxWidth - 1, 0],
            [maxWidth - 1, maxHeight - 1],
            [0, maxHeight - 1]], dtype="float32")

        # compute the perspective transform matrix and then apply it
        M = cv2.getPerspectiveTransform(rect, dst)
        warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

        # return the warped image
        return warped

    @staticmethod
    def apply_homography(src_points, dst_points, src_img, dst_img):
        # Calculate Homography
        h, status = cv2.findHomography(src_points, dst_points)

        # Warp source image to destination based on homography
        im_out = cv2.warpPerspective(src_img, h, (dst_img.shape[1], src_img.shape[0]))

        return im_out, h

    @staticmethod
    def point_to_point_homography(src_shot_location, h):
        # convert cartesian coordinates to homogeneous coordinates
        homo_coordinates = np.append(src_shot_location, [0])

        # do perspective transformation
        transformed_homo_coordinates = np.matmul(h, homo_coordinates)

        # convert homogeneous coordinates to cartesian coordinates
        transformed_cart_coordinates = np.array([int(transformed_homo_coordinates[0] / transformed_homo_coordinates[2]),
                                                 int(transformed_homo_coordinates[1] / transformed_homo_coordinates[2])])

        print(transformed_cart_coordinates)
        print(transformed_homo_coordinates)

        return transformed_cart_coordinates, transformed_homo_coordinates
