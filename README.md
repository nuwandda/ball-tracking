# Ball Tracking and Shot Location Estimation
A program that tracks the ball and estimates the shot location of a basketball shot.</br>  
  
  The program works in this way:</br>  
    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;1.Select 4 points for source destinations for homography. </br>
    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;2.Select ROI includes hoop. </br>
    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;3.Watch for a motion inside the ROI. </br>
    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;4.Validate the motion. With Blob Detection, distinguish the ball. </br>
    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;5.Track the ball 30-35 frames back to the beginning of the shot. </br>
    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;6.Find the shooter with YOLO. </br>
    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;7.Calculate homography between source image and destination image. </br>
    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;8.Convert world shot location points to the pixel shot location points with homography. </br>
    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;9.Mark the shot location on the heat map. </br>
    </br>
    
![alt text](https://github.com/nuwandda/ball-tracking/blob/development/screenshot01.png "Logo Title Text 1")

Thanks to the PyImageSearch for showing me the best examples and sources.
