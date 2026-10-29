
from pupil_apriltags import Detector as D
import cv2
import numpy as np
import os 
import tkinter as tk

# print(dir(cv2))
# print(dir(pupil_apriltags))

# image = cv2.imread
# QUAD_DECIMATE = 2.0  # for 4k? 

num_cores = os.cpu_count()
cv2.setNumThreads(num_cores)
print('Using', num_cores, 'cores')

# defines what tag is being detected
# specifically 36 data bits
# software compares the seen marker to the database

Tag = D(families="tag36h11", nthreads=num_cores)
# length of tag in Meters
tag_length = 0.1778 
# calibration distance (length of my laptop in meters)
cal_dis = 0.254
cal_txt = 'Press c when tag is 10 in from camera'


# Opens the camrea
camera = cv2.VideoCapture(0)
camera.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
camera.set(cv2.CAP_PROP_FRAME_WIDTH, 3840)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 2160)
camera.set(cv2.CAP_PROP_FPS, 30)

cam_w = int(camera.get(cv2.CAP_PROP_FRAME_WIDTH))
cam_h = int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
print('Camera resolution:', cam_w, 'x', cam_h)

text_scale = cam_w / 1280
big_font = 0.8 * text_scale
small_font = 0.6 * text_scale
text_thick = int(2 * text_scale)
line_thick = int(3 * text_scale)


root = tk.Tk()
screen_w = root.winfo_screenwidth()
screen_h = root.winfo_screenheight()
root.destroy()

scale_w = screen_w * 0.9 / cam_w
scale_h = screen_h * 0.9 / cam_h
if scale_w < scale_h:
    scale = scale_w
else:
    scale = scale_h
win_w = int(cam_w * scale)
win_h = int(cam_h * scale)

cv2.namedWindow('Webcam', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Webcam', win_w, win_h)


Running = True
Calibrated = False
while Running:
    # What key is being pressed (for 1 ms)
    key = cv2.waitKey(1) & 0xFF
    
    # it is capturing something (T/F) and the pixel size and color
    working , frame = camera.read()
    
    # wait 1 milisecond for key press and check if input key = character code of q
    if key == ord('q'):
        Running = False
        break
    if cv2.getWindowProperty('Webcam', cv2.WND_PROP_VISIBLE) < 1:
        Running = False
        break
    if not working:
        print('Cam not Working :(')
        break

    # changes color of specific frame from rgb 2 gray for contrast
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    
    # list of detected april tags
    detections = Tag.detect(gray)

                
    #Calibration Process
    # once camera is on we can find the pixel size of it
    if Calibrated == False:
        # press c when ready to calibrate & inclued prompt
        cv2.putText(frame,
                    # text
                    'Press c when tag within 1 ft',
                    # pos
                    (int(40 * text_scale), int(60 * text_scale)),
                    # font
                    cv2.FONT_HERSHEY_SIMPLEX,
                    # size
                    big_font,
                    # bgr color
                    color = (0, 0, 255),
                    # thickness
                    thickness= text_thick)
        if key == ord('c'):
            if detections:
                h, w = frame.shape[:2]
                cx = w/2
                cy = h/2
                cal_tag = detections[0]
                
                # camera parameters to determine focal distance
                corners = cal_tag.corners
                # top right - top left distance of marker
                top_w = np.linalg.norm(corners[1]-corners[0])
                bottom_w = np.linalg.norm(corners[2]-corners[3])
                avg_w = (top_w + bottom_w) / 2

                # focal length
                fx = avg_w*cal_dis/tag_length
                fy = fx  

                mycam_para = (fx,fy, cx, cy)
                Calibrated = True
            else:
                print('No tag detected for calibration')

    if Calibrated:
        # Get the center coordinates
        detections = Tag.detect(
            gray,
            estimate_tag_pose=True,
            camera_params=mycam_para,
            tag_size=tag_length
                )
   
    # list of coords for all tags 
    l_coords = []

    for tag in detections:

        # Convert corner coordinates to integers
        # & Draw the four sides of the tag
        corners = tag.corners.astype(int)

        for i in range(4):
            p1 = tuple(corners[i])
            p2 = tuple(corners[(i + 1) % 4])
            # make line from each corner on a frame with a certain bgr and thickness
            # Color_Bank = [(100, 0, 255),(255, 0, 100)]
            # Hi_Pryor = detections.index(tag) // 1
            cv2.line(frame, p1, p2,(0,0,255), line_thick)
        
        if Calibrated:
            
            center = tuple(tag.center.astype(int))

            # Pose estimation of each (position from center to cam center) by making it into a 1d array (x,y,z) in meters
            x, y, z = tag.pose_t.flatten()
            # x is left/right, y is up down, and z is away and towards
            l_coords.append(tag.pose_t.flatten())
            # offset and spacing
            b = int(30 * text_scale)
            m = int(25 * text_scale)
            # Write position in center of tag
            person_txt = f"Person: {tag.tag_id}:"
            cv2.putText(
                # on what v v
                frame,
                # people detection
                person_txt,
                # where
                (center[0]-b, center[1]+b),
                # font
                cv2.FONT_HERSHEY_SIMPLEX,
                # size
                small_font,
                # bgr color
                (0, 0, 255),
                # thickness
                text_thick
            )
            # for each orth direction
            for i in range(3):
                direction = ['X','Y','Z']
                offset =(2+i)*m
                xyz_txt = f"{direction[i]} ={tag.pose_t.flatten()[i]:.2f} m "
                cv2.putText(
                    frame,
                    xyz_txt,
                    (center[0]-b, center[1]+offset+b),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    small_font,
                    (0, 0, 255),
                    text_thick
                )

            if len(detections) >= 2:
                for i in range(1,len(detections)):
                    # must be a tuple of ints to make a line
                    cord1 = tuple( int(xy)  for xy in detections[i].center) 
                    cord2 = tuple( int(xy)  for xy in detections[i-1].center) 
                    cv2.arrowedLine(
                        frame,
                        cord1,
                        cord2,
                        (180, 218, 85),
                        line_thick
                    )
                    # to do the math
                
                # position of text in line
                cenline = tuple( ( np.array(cord1) + np.array(cord2) )//2  )
                # actual distance in m

                # print(type(detections))
                if detections.index(tag) > 0:

                    dx , dy, dz = tuple( np.array(l_coords[1])-np.array(l_coords[0]) )
                    #  magnitude
                    mag = np.sqrt( dx**2 + dy**2 + dz**2)
                    dist_txt = f"Total Distance ={mag:.2f} m "
                    cv2.putText(
                        frame,
                        dist_txt,
                        (cenline[0]-4*m, cenline[1]-b - 4*m),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        small_font,
                        (0, 0, 255),
                        text_thick
                    )

                    

        # show the frame
    cv2.imshow('Webcam',frame)


camera.release()
cv2.destroyAllWindows()
cv2.waitKey(1)
