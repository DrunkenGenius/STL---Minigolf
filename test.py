import cv2
import numpy as np;
import time
import math
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
import keyboard

#adasasdas

x=0
y=0

prevX=0
prevY=0
#gamle værdier
#StartX=229
#StartY=418
StartX=900
StartY=650

prevFrameTime = time.time()
currentTime = 0

posX = []
posY = []

isMoving = False
shots = 0
OB = False

maxVel = np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])



# ---------- Blob detecting function: returns keypoints and mask
# -- return keypoints, reversemask
def blob_detect(image,  # -- The frame (cv standard)
                hsv_min,  # -- minimum threshold of the hsv filter [h_min, s_min, v_min]
                hsv_max,  # -- maximum threshold of the hsv filter [h_max, s_max, v_max]
                blur=0,  # -- blur value (default 0)
                blob_params=None,  # -- blob parameters (default None)
                
                #Måske finde alt inden for en meget lille og lukket position omkring bolden
                search_window=None,
                # -- window where to search as [x_min, y_min, x_max, y_max] adimensional (0.0 to 1.0) starting from top left corner
                imshow=False
                ):
    # - Blur image to remove noise
    if blur > 0:
        image = cv2.blur(image, (blur, blur))
        # - Show result
        if imshow:
            cv2.imshow("Blur", image)
            cv2.waitKey(0)

    # - Search window
    if search_window is None: search_window = [0.0, 0.0, 1.0, 1.0]

    # - Convert image from BGR to HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # - Apply HSV threshold
    mask = cv2.inRange(hsv, hsv_min, hsv_max)

    # - Show HSV Mask
    if imshow:
        cv2.imshow("HSV Mask", mask)

    # - dilate makes the in range areas larger
    mask = cv2.dilate(mask, None, iterations=2)
    # - Show HSV Mask
    if imshow:
        cv2.imshow("Dilate Mask", mask)
        cv2.waitKey(0)

    mask = cv2.erode(mask, None, iterations=2)

    # - Show dilate/erode mask
    if imshow:
        cv2.imshow("Erode Mask", mask)
        cv2.waitKey(0)

    # - Cut the image using the search mask
    mask = apply_search_window(mask, search_window)

    if imshow:
        cv2.imshow("Searching Mask", mask)
        cv2.waitKey(0)

    # - build default blob detection parameters, if none have been provided
    if blob_params is None:
        # Set up the SimpleBlobdetector with default parameters.
        params = cv2.SimpleBlobDetector_Params()

        # Change thresholds
        params.minThreshold = 0;
        params.maxThreshold = 100;

        # Filter by Area.
        params.filterByArea = True
        #det som virker på den gamle
        #params.minArea = 30
        #params.maxArea = 20000
        params.minArea = 20
        params.maxArea = 200

        # Filter by Circularity
        params.filterByCircularity = True
        #Det gamle
        #params.minCircularity = 0.1
        params.minCircularity = 0.6

        # Filter by Convexity
        params.filterByConvexity = True
        params.minConvexity = 0.9
        #det gamle
        #params.minConvexity = 0.5

        # Filter by Inertia
        params.filterByInertia = True
        #det gamle
        #params.minInertiaRatio = 0.5
        params.minInertiaRatio = 0.5

    else:
        params = blob_params

        # - Apply blob detection
    detector = cv2.SimpleBlobDetector_create(params)

    # Reverse the mask: blobs are black on white
    reversemask = 255 - mask

    if imshow:
        cv2.imshow("Reverse Mask", reversemask)
        cv2.waitKey(0)

    keypoints = detector.detect(reversemask)

    return keypoints, reversemask


# ---------- Draw detected blobs: returns the image
# -- return(im_with_keypoints)
def draw_keypoints(image,  # -- Input image
                   keypoints,  # -- CV keypoints
                   line_color=(255, 0, 0),  # -- line's color (b,g,r)
                   imshow=False  # -- show the result
                   ):
    # -- Draw detected blobs as red circles.
    # -- cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures the size of the circle corresponds to the size of blob
    im_with_keypoints = cv2.drawKeypoints(image, keypoints, np.array([]), line_color,
                                          cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    #Til at beregne velocity
    global x,y
    global prevX, prevY
    #Til at beregne tid siden foregående frame altså deltaTime
    global prevFrameTime
    global currentTime

    global isMoving
    global shots
    global OB
    global StartX 
    global StartY 

    global maxVel
    


    #Sætter keypoints positions hvis der er nogen keypoints(altså den har detected en blob)
    if keypoints:
        x = keypoints[0].pt[0]
        y = keypoints[0].pt[1]
        calculateShots(x,y)
    else:
        calculateShots(x, y)

    #Her regnes deltaTime og prevFrameTime bliver sat
    currentTime = time.time()
    deltaTime = currentTime - prevFrameTime
    prevFrameTime = currentTime

    #Her regnes distancen blob har bevæget sig siden forrige frame
    speed = math.sqrt((prevX - x) ** 2 + (prevY - y) ** 2)
    np.append(maxVel, speed)
    maxVel = savgol_filter(maxVel, 11, 4)
    prevX, prevY = x,y

    #Start og slutpunkt for mål rektanglet på banen
    goalStartX, goalStartY = 1050,370
    goalEndX, goalEndY =1100, 420
    #Tekst med position og velocity af bold
    string = "x: " + str(int(x)) + " - y: " + str(int(y)) + " - velocity: "+str(round(speed,2))
    cv2.putText(im_with_keypoints, string, (50,300), cv2.FONT_HERSHEY_SIMPLEX, .5, (255,0,0), 2, cv2.LINE_AA)
    cv2.putText(im_with_keypoints, "Shots: " +  str(shots), (50, 250), cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 0, 0), 2, cv2.LINE_AA)
    cv2.rectangle(im_with_keypoints,(goalStartX,goalStartY),(goalEndX,goalEndY), (255,0,0), 2)
    #Hvis bolden er i målkassen
    if goalStartX < x < goalEndX and goalStartY < y < goalEndY:
        print("GOAL YAY")
    
    
    #Ydre kasse. Kassen er meget præcis og ser ud til at bolden skal forbi stregen før at det registreres som OB. med en nuværende goalStartY på 250, som er lige på kanten, ryger den ikke OB, når den rammer stregen.
    #den gamle der virker
    #ydreStartX, ydreStartY = 200,250    
    ydreStartX, ydreStartY = 70,500
    #ydreEndX værdi == 1000, hvis man vil teste OB
    #den gamle der virker
    #ydreEndX, ydreEndY =1200, 600
    ydreEndX, ydreEndY =950, 700
    cv2.rectangle(im_with_keypoints,(ydreStartX,ydreStartY),(ydreEndX,ydreEndY), (0,255,0), 2)
    
        
    if keyboard.is_pressed('r'):
        shots = 0
        
        #Virker ikke som det skal, den detecter pludseligt random ting uden for grænsen... tror dog stadigvæk ideen virker:"gradient" skal bare virke først
    #if (x<ydreStartX or ydreEndX<x or y<ydreStartY or ydreEndY<y) and OB == False:
       # print("OB")
        #OB = True
        #shots = shots +2
        
    #if ydreStartX < x < ydreEndX and ydreStartY < y < ydreEndY:
     #   print("In Bounds")
      #  OB = False
    
    if 2.5<speed < 20:
            isMoving = True
            
            
    
        
            #Behøves speed!= 0? her tjekkes om bolden er inden for rammerne og speed er under 1
    if speed < 1 and isMoving and ydreStartX < x < ydreEndX and ydreStartY < y < ydreEndY:
            print("In Bounds")
            OB = False
            #finder forige gange den lå stille og trækker det fra dens nuværende position, hvis den har flyttet sig mere end "x"px, så skal den tælle det som et skud
            if keypoints:
                x = keypoints[0].pt[0]
                y = keypoints[0].pt[1]
                print("x:"+str(x))
                print("y:"+str(y))
                #calculateShots(x,y)
                #vector3 and magnitude
                if abs(x-StartX)>50 or abs(y-StartY)>50:
                    shots = shots + 1
                    isMoving = False 
                    StartX = x
                    StartY = y
                else:
                    isMoving = False
                     # calculateShots(x, y)
           
              
                
        
    #Ydre kasseslut    
        
    if imshow:
        # Show keypoints
        cv2.imshow("Keypoints", im_with_keypoints)

    return (im_with_keypoints)
 

def calculateShots(x,y):
    #Lav array af positioner Man laver altid numpy array med korrekt størrelse(np.zero([n_frames])) udenfor for loop og "appender" med Array[Indeks], hvor Indeks f.eks er frame number. Np.append er mega langsom. Selv i real tid skal man bare allokere et stort array i hukommelsen, f.eks hvert minut
    global posX, posY
    posX.append(x)
    posY.append(y)

    # før du laver gradient på x og y array skal du lige bruge scipy filter for at fjerne outliers

    # gør position smooth
    #antal_frames = 11  # hvor mange frames der smoothes over
    #dfX = savgol_filter(posX, antal_frames, 4)
    #dfY = savgol_filter(posY, antal_frames, 4)

    # Lav gradient
    #V_x = np.gradient(dfX)
    #V_y = np.gradient(dfY)




# ---------- Draw search window: returns the image
# -- return(image)
def draw_window(image,  # - Input image
                window_adim,  # - window in adimensional units
                color=(255, 0, 0),  # - line's color
                line=5,  # - line's thickness
                imshow=False  # - show the image
                ):
    rows = image.shape[0]
    cols = image.shape[1]

    x_min_px = int(cols * window_adim[0])
    y_min_px = int(rows * window_adim[1])
    x_max_px = int(cols * window_adim[2])
    y_max_px = int(rows * window_adim[3])

    # -- Draw a rectangle from top left to bottom right corner
    image = cv2.rectangle(image, (x_min_px, y_min_px), (x_max_px, y_max_px), color, line)

    if imshow:
        # Show keypoints
        cv2.imshow("Keypoints", image)

    return (image)


# ---------- Draw X Y frame
# -- return(image)
def draw_frame(image,
               dimension=0.3,  # - dimension relative to frame size
               line=2  # - line's thickness
               ):
    rows = image.shape[0]
    cols = image.shape[1]
    size = min([rows, cols])
    center_x = int(cols / 2.0)
    center_y = int(rows / 2.0)

    line_length = int(size * dimension)

    # -- X
    image = cv2.line(image, (center_x, center_y), (center_x + line_length, center_y), (0, 0, 255), line)
    # -- Y
    image = cv2.line(image, (center_x, center_y), (center_x, center_y + line_length), (0, 255, 0), line)

    return (image)


# ---------- Apply search window: returns the image
# -- return(image)
def apply_search_window(image, window_adim=[0.0, 0.0, 1.0, 1.0]):
    rows = image.shape[0]
    cols = image.shape[1]

    searchMinX = 70
    searchMinY = 500
    searchMaxX = 950
    searchMaxY = 700
    
    #searchMinX = 200
    #searchMinY = 250
    #searchMaxX = 1200
    #searchMaxY = 600

    x_min_px = searchMinX
    y_min_px = searchMinY
    x_max_px = searchMaxX
    y_max_px = searchMaxY

    # --- Initialize the mask as a black image
    mask = np.zeros(image.shape, np.uint8)

    # --- Copy the pixels from the original image corresponding to the window
    mask[y_min_px:y_max_px, x_min_px:x_max_px] = image[y_min_px:y_max_px, x_min_px:x_max_px]

    # --- return the mask
    return (mask)


# ---------- Apply a blur to the outside search region
# -- return(image)
def blur_outside(image, blur=5, window_adim=[0.0, 0.0, 1.0, 1.0]):
    rows = image.shape[0]
    cols = image.shape[1]
    x_min_px = int(cols * window_adim[0])
    y_min_px = int(rows * window_adim[1])
    x_max_px = int(cols * window_adim[2])
    y_max_px = int(rows * window_adim[3])

    # --- Initialize the mask as a black image
    mask = cv2.blur(image, (blur, blur))

    # --- Copy the pixels from the original image corresponding to the window
    mask[y_min_px:y_max_px, x_min_px:x_max_px] = image[y_min_px:y_max_px, x_min_px:x_max_px]

    # --- return the mask
    return (mask)


# ---------- Obtain the camera relative frame coordinate of one single keypoint
# -- return(x,y)
def get_blob_relative_position(image, keyPoint):
    rows = float(image.shape[0])
    cols = float(image.shape[1])
    # print(rows, cols)
    center_x = 0.5 * cols
    center_y = 0.5 * rows
    # print(center_x)
    x = (keyPoint.pt[0] - center_x) / (center_x)
    y = (keyPoint.pt[1] - center_y) / (center_y)
    return (x, y)


# ----------- TEST
if __name__ == "__main__":
    globals()

    # --- Define HSV limits
    blue_min = (51, 93, 65)
    blue_max = (58, 166, 124)
    
    # værdier der virker på video i vores lab
    #blue_min = (0, 178, 145)
    #blue_max = (183, 255, 255)
    #nye værdier
     #0,162,117
    #110,218,255
    
    #De gamle værdier til den gamle video
    #blue_min = (0, 173, 171)
    #blue_max = (7, 255, 255)

    # --- Define area limit [x_min, y_min, x_max, y_max] adimensional (0.0 to 1.0) starting from top left corner
    window = [0, 0, 1, 1]

    # -- IMAGE_SOURCE: either 'camera' or 'imagelist'
    # SOURCE = 'video'
    SOURCE = 'video'

    if SOURCE == 'video':
        #den der virker i vores "lab"
        #cap = cv2.VideoCapture("nyVideoMedStilleKamera.mov") 
        cap = cv2.VideoCapture("NormaltForsøg1.mp4")
        while (True):
            # Capture frame-by-frame
            ret, frame = cap.read()

            
            frameResize = cv2.resize(frame, dsize=(int(frame.shape[1]*1.5),int(frame.shape[0]*1.5)))
            #den der virker på den gamle video
            #frameResize = cv2.resize(frame, dsize=(int(frame.shape[1]*90/100),int(frame.shape[0]*90/100)))

            # -- Detect keypoints
            keypoints, _ = blob_detect(frameResize, blue_min, blue_max, blur=3,
                                       blob_params=None, search_window=window, imshow=False)
            # -- Draw search window
            frameResize = draw_window(frameResize, window)

            # -- click ENTER on the image window to proceed
            draw_keypoints(frameResize, keypoints, imshow=True)

            # -- press q to quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    else:
        # -- Read image list from file:
        image_list = []
        image_list.append(cv2.imread("golf.JPG"))
        # image_list.append(cv2.imread("blob2.jpg"))
        # image_list.append(cv2.imread("blob3.jpg"))

        for image in image_list:
            # -- Detect keypoints
            keypoints, _ = blob_detect(image, blue_min, blue_max, blur=5,
                                       blob_params=None, search_window=window, imshow=True)

            image = blur_outside(image, blur=15, window_adim=window)
            cv2.imshow("Outside Blur", image)
            cv2.waitKey(0)

            image = draw_window(image, window, imshow=True)
            # -- enter to proceed
            cv2.waitKey(0)

            # -- click ENTER on the image window to proceed
            image = draw_keypoints(image, keypoints, imshow=True)
            cv2.waitKey(0)
            # -- Draw search window

            image = draw_frame(image)
            cv2.imshow("Frame", image)
            cv2.waitKey(0)

