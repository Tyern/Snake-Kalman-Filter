import os
import glob
import cv2
from snakeDetector import detect
from KalmanFilter import KalmanFilter2 as KalmanFilter
import numpy as np
import matplotlib.pyplot as plt
import copy
import shutil

def main():
    image_dir = "image"
    if os.path.isdir(image_dir):
        shutil.rmtree(image_dir)
    os.makedirs(image_dir)
        
    # Create opencv video capture object
    VideoCap = cv2.VideoCapture('video/snake_auf_3.mp4')

    #Variable used to control the speed of reading the video
    ControlSpeedVar = 100  #Lowest: 1 - Highest:100

    HiSpeed = 100

    #Create KalmanFilter object KF
    #KalmanFilter(dt, u_x, u_y, std_acc, x_std_meas, y_std_meas)

    KF = KalmanFilter(5, 0, 0, 1, 5, 5)

    debugMode=1
    
    accurate_pos = []
    kalman_pred_res = []
    kalman_est_res = []
    img_size = None
    ret = True
    i = 0
    centers = None
    
    while True:
        # Read frame
        ret, frame = VideoCap.read()
        if not ret:
            break
        
        img_size = img_size or frame.shape

        old_center = copy.deepcopy(centers)
        # Detect object
        flag, centers = detect(frame)
        centers = centers if flag else old_center

        # If centroids are detected then track them
        if (flag):
            accurate_pos.append(centers)

            # Draw the detected circle
            cv2.circle(frame, (int(centers[0]), int(centers[1])), 10, (0, 191, 255), 2)
            cv2.putText(frame, "Traditional method", (int(centers[0] + 15), int(centers[1] - 15)), 0, 0.5, (0,191,255), 2)
        
        # Predict
        (x, y) = KF.predict()
        kalman_pred_res.append((x, y))
        # Draw a rectangle as the predicted object position
        if (not flag):
            kalman_est_res.append((x, y))
            cv2.rectangle(frame, (int(x - 15), int(y - 15)), (int(x + 15), int(y + 15)), (0, 0, 255), 2)
            cv2.putText(frame, "Kalman estimated method", (int(x + 15), int(y + 10)), 0, 0.5, (0, 0, 255), 2)

        if (flag):
            # Update
            (x1, y1) = KF.update(centers)
            kalman_est_res.append((x1, y1))

            # Draw a rectangle as the estimated object position
            cv2.rectangle(frame, (int(x1 - 15), int(y1 - 15)), (int(x1 + 15), int(y1 + 15)), (0, 0, 255), 2)
            cv2.putText(frame, "Kalman estimated method", (int(x1 + 15), int(y1 + 10)), 0, 0.5, (0, 0, 255), 2)

        cv2.imwrite(os.path.join(image_dir, f"{i:0>6}.jpg"), frame)
        i += 1
        
        # cv2.imshow('image', frame)
        # if cv2.waitKey(2) & 0xFF == ord('q'):
        #     VideoCap.release()
        #     cv2.destroyAllWindows()
        #     break
        # cv2.waitKey(HiSpeed-ControlSpeedVar+1)
        
    VideoCap.release()
    cv2.destroyAllWindows()
    image_glob = sorted(glob.glob(os.path.join(image_dir, "*.jpg")))
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
    video = cv2.VideoWriter('video/result.mp4', fourcc, 12, (img_size[1], img_size[0]))
    
    for image_path in image_glob:
        video.write(cv2.imread(image_path))
    video.release()
    
    cv2.destroyAllWindows()
        
    accurate_pos = np.array(accurate_pos).squeeze()
    kalman_est_res = np.array(kalman_est_res).squeeze()

    plt.plot(*zip(*accurate_pos), label="Traditional method", color='y',linewidth=0.5)
    plt.plot(*zip(*kalman_est_res), label="Kalman estimated method", color='r',linewidth=0.5)
    plt.scatter(*zip(*accurate_pos), s=5, color='y', marker="o", alpha=0.4)
    plt.scatter(*zip(*kalman_est_res), s=5, color='r', marker="o", alpha=0.4)
    
    plt.legend()
    plt.gca().invert_yaxis()
    plt.savefig("path.jpg", dpi=1000)
    
    plt.show()
    


if __name__ == "__main__":
    # execute main
    main()
