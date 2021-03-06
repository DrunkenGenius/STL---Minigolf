import cv2
import numpy as np;
import math
from scipy.signal import savgol_filter
import keyboard
import time

#Global Variables
x=0
y=0
prevX=0
prevY=0
StartX=229
StartY=418
posX = []
posY = []
isMoving = False
shots = 0
OB = False
score = 0
#Bruges ikke, men kan bruges til at lave en graf og finde outliers
maxVel = np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
goalReset = False

# ---------- Blob detecting function: returns keypoints and mask
# -- return keypoints, reversemask
def blob_detect(image,  # -- The frame (cv standard)
                hsv_min,  # -- minimum threshold of the hsv filter [h_min, s_min, v_min]
                hsv_max,  # -- maximum threshold of the hsv filter [h_max, s_max, v_max]
                blur=0,  # -- blur value (default 0)
                blob_params=None,  # -- blob parameters (default None)
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
        cv2.waitKey(0)

    # - dilate makes the in range areas larger
    #Det udvider det som er inde i HSV-filteret, iterations betyder at den gør det 2 gange
    mask = cv2.dilate(mask, None, iterations=2)
    # - Show HSV Mask
    if imshow:
        cv2.imshow("Dilate Mask", mask)
        cv2.waitKey(0)

    mask = cv2.erode(mask, None, iterations=2)

    # - Show dilate/erode mask
    #Gør det modsatte af dilate
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
        #opretter en object med default parametre. Pointen er her at vi så kan ændre i værdierne, så det passer til vores video.
        params = cv2.SimpleBlobDetector_Params()

        # Change thresholds
        params.minThreshold = 0;
        params.maxThreshold = 100;

        # Filter by Area.
        params.filterByArea = True
        params.minArea = 30
        params.maxArea = 200000

        # Filter by Circularity
        params.filterByCircularity = True
        params.minCircularity = 0.1

        # Filter by Convexity
        params.filterByConvexity = True
        params.minConvexity = 0.5

        # Filter by Inertia
        params.filterByInertia = True
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

    return keypoints

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

    calculate_shots(im_with_keypoints, keypoints)

    #den viser den nuværende fram i videon vi ser
    if imshow:
        # Show keypoints
        cv2.imshow("Keypoints", im_with_keypoints)
    return (im_with_keypoints)


def calculate_shots(im_with_keypoints, keypoints):
    # Til at beregne velocity
    global x, y, prevX, prevY, isMoving, shots, OB, StartX, StartY, maxVel, goalReset
    #Parametre
    minMoveDistance = 20
    goalStartX, goalStartY = 490, 335
    goalEndX, goalEndY = goalStartX + 20, goalStartY + 20

    # Sætter keypoints positions hvis der er nogen keypoints(altså den har detected en blob)
    if keypoints:
        x = keypoints[0].pt[0]
        y = keypoints[0].pt[1]

    # calcMagnitude kalkulerer magnitude for to punkter.
    speed = calcMagnitude(prevX, prevY, x, y)

    # Her vil vi prøve at lave en kurve, så vi fjerner outliers
    #GØR IKKE NOGET PT
    np.append(maxVel, speed)
    maxVel = savgol_filter(maxVel, 11, 4)
    prevX, prevY = x, y

    # Laver mål ramme og sørger for hvad der skal ske hvis der er mål i funktionen handleGoal()
    createGoal(im_with_keypoints, goalStartX, goalStartY, goalEndX, goalEndY, x, y)

    # Ydre kasse. Kassen er meget præcis og ser ud til at bolden skal forbi stregen før at det registreres som OB(Out of Bounds)
    ydreStartX, ydreStartY = 10, 10
    #ydreEndX værdi == 1000, hvis man vil teste OB
    ydreEndX, ydreEndY = 1260, 700

    #Viser out of bounds kassen
    #cv2.rectangle(im_with_keypoints,(ydreStartX,ydreStartY),(ydreEndX,ydreEndY), (0,255,0), 2)

    if keyboard.is_pressed('r'):
        shots = 0

    check_OB()

    if 1.5 < speed < 20:
        isMoving = True
    if speed < 1 and isMoving and ydreStartX < x < ydreEndX and ydreStartY < y < ydreEndY:
        print("In Bounds")
        OB = False
        # finder forige gange den lå stille og trækker det fra dens nuværende position, hvis den har flyttet sig mere end "x"px, så skal den tælle det som et skud
        if keypoints:
            x = keypoints[0].pt[0]
            y = keypoints[0].pt[1]
            # her bruger vi magnitude til at lave en cirkel rundt om boldens sidste position.
            # Hvis bolden kommer ud over den cirkel, så tæller den et nyt skud
            magnitude = calcMagnitude(StartX, StartY, x, y)
            if magnitude > minMoveDistance:
                shots = shots + 1
                isMoving = False
                StartX = x
                StartY = y
            else:
                isMoving = False

    printInfo(im_with_keypoints, shots, speed, x, y)


def check_OB():
    pass
    # Virker ikke som det skal, den detecter pludseligt random ting uden for grænsen... tror dog stadigvæk ideen virker:"gradient" skal bare virke først
    # problemet er lige nu at der er outliers
    # if (x<ydreStartX or ydreEndX<x or y<ydreStartY or ydreEndY<y) and OB == False:
    # print("OB")
    # OB = True
    # shots = shots +2
    # if ydreStartX < x < ydreEndX and ydreStartY < y < ydreEndY:
    # print("In Bounds")
    # OB = False


def printInfo(im_with_keypoints, shots, speed, x, y):
    # Tekst med position og velocity af bold
    position = "x: " + str(int(x)) + " - y: " + str(int(y)) + " - velocity: " + str(round(speed, 2))
    cv2.putText(im_with_keypoints, position, (50, 300), cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(im_with_keypoints, "Shots: " + str(shots), (50, 250), cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 0, 0), 2,
                cv2.LINE_AA)
    #cv2.putText(im_with_keypoints, "Score: " + str(score), (50, 275), cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 0, 0), 2,
    #            cv2.LINE_AA)


def calcMagnitude(StartX, StartY, x, y):
    #Pythagoras for at finde længden mellem to punkter
    return math.sqrt((StartX - x) ** 2 + (StartY - y) ** 2)

def createGoal(im_with_keypoints, goalStartX, goalStartY, goalEndX, goalEndY, x,y):
    # Start og slutpunkt for mål rektanglet på banen.
    # Skal sættes til det nuværende billede, da det pt. er uden for billedet
    cv2.rectangle(im_with_keypoints, (goalStartX, goalStartY), (goalEndX, goalEndY), (255, 0, 0), 2)

    if goalStartX < x < goalEndX and goalStartY < y < goalEndY:
        handleGoal()


def handleGoal():
    global shots, score
    # Hvis bolden er i målkassen
    # Dette ville skulle bruges for at registrere at bolden er i hul og at den person der spiller er færdig.
    shotPlusOne = shots + 1
    if shotPlusOne ==1:
        score = 1000
    if shotPlusOne ==2:
        score = 800
    if shotPlusOne == 3:
        score = 600
    if shotPlusOne ==4:
        score = 400
    if shotPlusOne ==5:
        score = 200
    if shotPlusOne ==6:
        score = 100


# ---------- Draw search window: returns the image
# -- return(image)
#den tegner vores mask, så vi visuelt kan se hvilket felt vi søger inden for.
def draw_window(image,  # - Input image
                window_adim,  # - window in adimensional units
                color=(0, 255, 0),  # - line's color
                line=5,  # - line's thickness
                imshow=False  # - show the image
                ):
    rows = image.shape[0]
    cols = image.shape[1]
    #vinduet er sat nede i main og hedder window 
    x_min_px = int(cols * window_adim[0])
    y_min_px = int(rows * window_adim[1])
    x_max_px = int(cols * window_adim[2])
    y_max_px = int(rows * window_adim[3])

    # -- Draw a rectangle from top left to bottom right corner
    image = cv2.rectangle(image, (x_min_px, y_min_px), (x_max_px, y_max_px), color, line)
    return (image)

# ---------- Apply search window: returns the image
# -- return(image)
#funktionen gør at vi ikke søger i det modsatte område af hvad vi angiver i "window"
def apply_search_window(image, window_adim=[0.0, 0.0, 1.0, 1.0]):
    rows = image.shape[0]
    cols = image.shape[1]
	#søger indefor dette areal
    x_min_px = int(window_adim[0]*cols)
    y_min_px = int(window_adim[1]*rows)
    x_max_px = int(window_adim[2]*cols)
    y_max_px = int(window_adim[3]*rows)

    # --- Initialize the mask as a black image
    mask = np.zeros(image.shape, np.uint8)

    # --- Copy the pixels from the original image corresponding to the window
    mask[y_min_px:y_max_px, x_min_px:x_max_px] = image[y_min_px:y_max_px, x_min_px:x_max_px]

    # --- return the mask
    return (mask)

#Den kører denne if sætning, fordi det er det her program vi kører. Havde vi impoteret denne fil, så ville den ikke køre nedestående kode
if __name__ == "__main__":
    #hsv_min,  # -- minimum threshold of the hsv filter [h_min, s_min, v_min]
    # --- Define HSV limits
    #Her skal man sætte sine parametre man har fra rangedetectoren.
    #koden for at køre rangedetectoren er: python range-detector.py --image forsog1MedLys.png --filter HSV --preview
    red_min=(0,150,57)
    red_max=(33,255,255)


    # --- Define area limit [x_min, y_min, x_max, y_max] adimensional (0.0 to 1.0) starting from top left corner
    #Her sætter vi det vindue vi gerne vil søge inden for
    window = [0, 0.5, 1, 1]

    # -- IMAGE_SOURCE: either 'camera' or 'imagelist'
    # SOURCE = 'video'
    SOURCE = 'video'

    if SOURCE == 'video':
								#"nyVideoMedStilleKamera.mov"
        cap = cv2.VideoCapture("Bane1.avi")
        while (True):
            # Capture frame-by-frame
            # ret er en bool der siger om noget er returned
            ret, frame = cap.read()

            #
            frameResize = cv2.resize(frame, dsize=(int(frame.shape[1]*90/100),int(frame.shape[0]*90/100)))

            # -- Detect keypoints
            #Blur parameteren kan ændres og har stor betydning for hvordan vi finder bolden, man kan ændre i denne værdi for at finde det billede der tracker bolden bedst
            keypoints= blob_detect(frameResize, red_min, red_max, blur=3,
                                       blob_params=None, search_window=window, imshow=False)
            # -- Draw search window
            frameResize = draw_window(frameResize, window)

            # -- click ENTER on the image window to proceed
            draw_keypoints(frameResize, keypoints, imshow=True)
            time.sleep(0.025)
            # -- press q to quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
cv2.destroyAllWindows()


