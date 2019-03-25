class FrameObject:
    """An object to hold frame's index and actual frame
    
    Returns:
        frame -- Actual frame of the image

        index -- Index of the frame
    """
    def __init__(self,frame = None,index = None):
        self.frame = frame
        self.index = index

    def getFrame(self):
        return self.frame

    def getIndex(self):
        return self.index