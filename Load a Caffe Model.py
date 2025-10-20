import cv2

# Load a Caffe Model
protoFile = "./mpi.prototxt"
weightsFile = "./pose_iter_160000.caffemodel"
# Specify number of points in the model
nPoints = 15
POSE_PAIRS = [[0, 1], [1, 2], [2, 3], [3, 4], [1, 5], [5, 6], [6, 7], [1, 14], [14, 8], [8, 9], [9, 10], [14, 11], [11, 12], [12, 13]]
net = cv2.dnn.readNetFromCaffe(protoFile, weightsFile)

# Confidence treshold
threshold = 0.1

##########################################################################
def Drawskeleton(POSE_PAIRS, im):
    # Draw skeleton
    imSkeleton = im.copy()
    for pair in POSE_PAIRS:
        partA = pair[0]
        partB = pair[1]
        if points[partA] and points[partB]:
            cv2.line(imSkeleton, points[partA], points[partB], (255, 255, 0), 2)
            cv2.circle(imSkeleton, points[partA], 8, (255, 0, 0), thickness=-1, lineType=cv2.FILLED)
    cv2.imshow("Skeleton", imSkeleton)
##########################################################################
def DrawPoints(points, im):
    # Draw points
    imPoints = im.copy()
    for i, p in enumerate(points):
        cv2.circle(imPoints, p, 8, (255, 255, 0), thickness=-1, lineType=cv2.FILLED)
        cv2.putText(imPoints, "{}".format(i), p, cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, lineType=cv2.LINE_AA)
    cv2.imshow("Keypoints", imPoints)
##########################################################################

## camera init
cap = cv2.VideoCapture(0)
ret, _ = cap.read()
if (ret == False):
    print("ERRO: falta camara")
    exit(-1)
# pose detection and action
while True:
    # get frame from the video
    hasFrame, im = cap.read()

    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    inWidth = im.shape[1]
    inHeight = im.shape[0]

    # Convert image to blob
    netInputSize = (368, 368)
    inpBlob = cv2.dnn.blobFromImage(im, 1.0 / 255, netInputSize, (0, 0, 0), swapRB=True, crop=False)
    net.setInput(inpBlob)

    # Run Inference (forward pass)
    output = net.forward()

    # Extract points: X and Y Scale
    scaleX = float(inWidth) / output.shape[3]
    scaleY = float(inHeight) / output.shape[2]

    # Empty list to store the detected keypoints
    points = []



    for i in range(nPoints):
        # Obtain probability map
        probMap = output[0, i, :, :]

        # Find global maxima of the probMap.
        minVal, prob, minLoc, point = cv2.minMaxLoc(probMap)

        # Scale the point to fit on the original image
        x = scaleX * point[0]
        y = scaleY * point[1]

        if prob > threshold:
            # Add the point to the list if the probability is greater than the threshold
            points.append((int(x), int(y)))
        else:
            points.append(None)

    # show points coordinates
    # for i in range(nPoints):
    #     if points[i] is not None:
    #         print("pointnum:", i, "Xcoord", points[i][0], "Ycoord", points[i][1])
    #     else:
    #         print("pointnum:", i, "not detected")

    if points[4] == None or points[1] == None:
        print("4 nao detetado")
    elif points[4][1] < points[1][1]:
        print("Braço Direito no ar", points[4][1], points[1][1])

    if points[4] == None or points[1] == None or points[7] == None or points[1] == None:
        print("4 e 7 não detetados")
    elif points[4][1] < points[1][1] and points[7][1] < points[1][1]:
        print("Os dois braços no ar")

    if points[1] == None or points[2] == None or points[4] == None:
        print("1, 2 e 4 não detetados")
    elif points[4][0] > points[2][0]:
        print("Grande movimento de right to left")
    elif points[4][0] > points[1][0]:
        print("Pequeno movimento de right to left")

    if points[1] == None or points[2] == None or points[7] == None:
        print("1, 2 e 7 não detetados")
    elif points[7][0] < points[2][0]:
        print("Grande movimento de left to right")
    elif points[7][0] < points[1][0]:
        print("Pequeno movimento de left to right")


    # ONLY For Display Points & Skeleton
    DrawPoints(points, im)
    Drawskeleton(POSE_PAIRS, im)

    if cv2.waitKey(10) == 27:
        break

cv2.destroyAllWindows()