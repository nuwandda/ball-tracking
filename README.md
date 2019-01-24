# ball-tracking
A ball tracking program that detects the motion and distinguish the ball with Blob Detection.</br>  
  
  The program works in this way:</br>  
    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;1.Get the first frame. </br>
    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;2.Convert it to gray. </br>
    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;3.Apply Background Subtraction Apply opening and closing. </br>
    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;4.Dilate the current frame. </br>
    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;5.Find contours on the local area. </br>
    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;6.Use Blob Detection to find the ball </br>
    </br>
    
![alt text](https://github.com/nuwandda/ball-tracking/blob/development/screenshot01.png "Logo Title Text 1")
