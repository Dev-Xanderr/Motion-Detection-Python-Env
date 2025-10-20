import cv2
import time
MODEL_PATH = "SSD_Faces/"
imgpessoa = cv2.imread('ornn.jpg')

conf_threshold = 0.7
cap = cv2.VideoCapture(0)
hasFrame, frame = cap.read()
frame_count = 0
tt_opencvDnn = 0

def detectFaceOpenCVDnn(net, frame):
    frameOpencvDnn = frame.copy()
    frameHeight = frameOpencvDnn.shape[0]
    frameWidth = frameOpencvDnn.shape[1]
    blob = cv2.dnn.blobFromImage(frameOpencvDnn, 1.0, (300, 300), [104, 117, 123], False, False)

    net.setInput(blob)
    detections = net.forward()
    bboxes = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_threshold:
            x1 = int(detections[0, 0, i, 3] * frameWidth)
            y1 = int(detections[0, 0, i, 4] * frameHeight)
            x2 = int(detections[0, 0, i, 5] * frameWidth)
            y2 = int(detections[0, 0, i, 6] * frameHeight)
            bboxes.append([x1, y1, x2, y2])
            cv2.rectangle(frameOpencvDnn, (x1, y1), (x2, y2), (0, 255, 0), int(round(frameHeight/150)), 8)

            wPRoi = x2 - x1
            hPRoi = y2 - y1

            imgPRed = cv2.resize(imgpessoa, (wPRoi, hPRoi), interpolation=cv2.INTER_AREA)
            if (y1 > 0 and y1 + hPRoi > 0 and x1 > 0 and x1 + wPRoi > 0 and y1 < frameHeight and y1 + hPRoi < frameHeight and x1 < frameWidth and x1 + wPRoi < frameWidth):
                frameOpencvDnn[y1:y1 + hPRoi, x1:x1 + wPRoi] = imgPRed
    return frameOpencvDnn, bboxes



DNN = "TF"
if DNN == "CAFFE": # 1. CAFEE - FP16 version of the original caffe implementation ( 5.4 MB )
    modelFile = MODEL_PATH + "res10_300x300_ssd_iter_140000_fp16.caffemodel"
    configFile = MODEL_PATH + "deploy.prototxt"
    net = cv2.dnn.readNetFromCaffe(configFile, modelFile)
else: # 2. TF - 8 bit Quantized version using Tensorflow ( 2.7 MB )
    modelFile = MODEL_PATH + "opencv_face_detector_uint8.pb"
    configFile = MODEL_PATH + "opencv_face_detector.pbtxt"
    net = cv2.dnn.readNetFromTensorflow(modelFile, configFile)




while(True):
    hasFrame, frame = cap.read()
    if not hasFrame:
        break
    frame_count += 1
    t = time.time()
    outOpencvDnn, bboxes = detectFaceOpenCVDnn(net,frame)
    tt_opencvDnn += time.time() - t
    fpsOpencvDnn = frame_count / tt_opencvDnn
    label = "OpenCV DNN ; FPS : {:.2f}".format(fpsOpencvDnn)


    cv2.putText(outOpencvDnn, label, (10,50), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 0, 255), 3, cv2.LINE_AA)
    cv2.imshow("Face Detection Comparison", outOpencvDnn)
    if frame_count == 1:
        tt_opencvDnn = 0
    k = cv2.waitKey(10)
    if k == 27:
        break
