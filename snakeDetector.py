import cv2
import numpy as np


def detect(frame):
    """
    Detect the position of a pattern image within a larger image,
    allowing the pattern to rotate in four directions (0°, 90°, 180°, 270°).

    Parameters:
        pat_img (numpy.ndarray): The pattern image to detect.
        frame (numpy.ndarray): The larger image to search within.

    Returns:
        tuple: A tuple containing the center coordinates of the detected pattern 
               and the rotation angle, or None if no match is found.
    """
    # Convert images to grayscale
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
    
    threshold_value = 230
    max_value = 255

    # Apply a binary threshold
    _, frame_gray = cv2.threshold(frame_gray, threshold_value, max_value, cv2.THRESH_BINARY)
    
    # Find contours
    contours, _ = cv2.findContours(frame_gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Set the accepted minimum & maximum radius of a detected object
    min_radius_thresh= 2
    max_radius_thresh= 30

    centers=[]
    for c in contours:
        # ref: https://docs.opencv.org/trunk/dd/d49/tutorial_py_contour_features.html
        (x, y), radius = cv2.minEnclosingCircle(c)
        if y < 50:
            continue
        radius = int(radius)

        #Take only the valid circle(s)
        if (radius > min_radius_thresh) and (radius < max_radius_thresh):
            centers.append(np.array([[x], [y]]))
            
    flag = False
    if centers:
        centers = np.array(centers).mean(axis=0)
        centers = centers + np.random.randn(2, 1)
        flag = True
    else:
        centers = None
        flag = False
        
    return flag, centers
    

if __name__ == "__main__":
    # Example usage:
    VideoCap = cv2.VideoCapture('video/snake_auf_1.mp4')

    while True:
        # Read frame
        ret, frame = VideoCap.read()
        if not ret:
            break
        
        flag, centers = detect(frame)
        if flag:
            cv2.circle(frame, (int(centers[0]), int(centers[1])), 10, (255,0, 0), 4)

        cv2.imshow('image', frame)
        if cv2.waitKey(2) & 0xFF == ord('q'):
            VideoCap.release()
            cv2.destroyAllWindows()
            break

        cv2.waitKey(0)
        