import cv2


class DetectBall:
    """Detects ball with Blob Detection

    Returns:
        List -- Returns list of key points
    """

    def __init__(self):
        pass

    def detectBall(image):
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
