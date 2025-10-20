import numpy as np
import cv2



ThresMov = 4500000

cap = cv2.VideoCapture(0)
ret, frame = cap.read()
if (ret == False):
    print("ERRO: falta camara")
    exit(-1)
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

# parâmetros obrigatórios para funcioanr
threshold = 0.7
inWidth = 300
inHeight = 300
mean = (127.5, 127.5, 127.5)

modelFile = "ssdMobilenetV2Coco20180329inference.pb"
configFile = "ssd_mobilenet_v2_coco_2018_03_29.pbtxt"
classFile = "coco_class_labels.txt"


imagePath = cv2.imread("ornn.jpg")
imagePath2 = cv2.imread("monitor.jpg")
imagePath3 = cv2.imread("teclado.jpeg")


with open(classFile) as fi:
    labels = fi.read().split('\n')
    # print (labels)
net = cv2.dnn.readNetFromTensorflow(modelFile, configFile)
ii = 0

ret, oFrame = cap.read()
oFrameG = cv2.cvtColor(oFrame, cv2.COLOR_BGR2GRAY)
h, w, c = oFrame.shape


while (cap.isOpened()):

    ret, frame = cap.read()

    rows = frame_height
    cols = frame_width
    ii += 1
    net.setInput(cv2.dnn.blobFromImage(frame, 1.0 / 127.5, (inWidth, inHeight), mean, True, False))
    out = net.forward()

    for i in range(out.shape[2]):
        score = float(out[0, 0, i, 2])
        classId = int(out[0, 0, i, 1])
        x1 = int(out[0, 0, i, 3] * cols)
        y1 = int(out[0, 0, i, 4] * rows)
        x2 = int(out[0, 0, i, 5] * cols)
        y2 = int(out[0, 0, i, 6] * rows)
        if score > threshold:
            # cv2.putText(frame, "Object : {}, confidence = {:.3f}".format(labels[classId], score), ( x1, y1 - 10 ), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
            # 72-tv 76-moyse 73-keyboard

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), cv2.FONT_HERSHEY_DUPLEX, 4)
            frameG = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            resultadoT = cv2.subtract(frameG, oFrameG)
            ret, resultado = cv2.threshold(resultadoT, 100, 255, cv2.THRESH_BINARY)
            resSum = np.sum(resultado[:, :])

            if (resSum > ThresMov):
                print("Movimento")
                if classId == 1:
                    wimg1 = x2 - x1
                    himg1 = y2 - y1

                    res1 = cv2.resize(imagePath, (wimg1, himg1), interpolation=cv2.INTER_CUBIC)

                    if (y1 > 0 and y1 + himg1  > 0 and x1 > 0 and x1 + wimg1 > 0 and y1 < h and y1 + himg1  < h and x1 < w and x1 + wimg1 < w):
                        frame[y1:y1 + himg1, x1:x1 + wimg1] = res1

                if classId == 72:
                    wimg2 = x2 - x1
                    himg2 = y2 - y1

                    res2 = cv2.resize(imagePath2, (wimg2, himg2), interpolation=cv2.INTER_CUBIC)

                    if (y1 > 0 and y1 + himg2  > 0 and x1 > 0 and x1 + wimg2 > 0 and y1 < h and y1 + himg2  < h and x1 < w and x1 + wimg2 < w):
                        frame[y1:y1 + himg2, x1:x1 + wimg2] = res2

                if classId == 76:
                    wimg3 = x2 - x1
                    himg3 = y2 - y1

                    res3 = cv2.resize(imagePath3, (wimg3, himg3), interpolation=cv2.INTER_CUBIC)

                    if (y1 > 0 and y1 + himg3 > 0 and x1 > 0 and x1 + wimg3 > 0 and y1 < h and y1 + himg3 < h and x1 < w and x1 + wimg3 < w):
                        frame[y1:y1 + himg3, x1:x1 + wimg3] = res3

            print(labels[classId], score)



    cv2.imshow("OpenCV Tensorflow SSD", frame)
    k = cv2.waitKey(10)
    if k == 27:
        break
cv2.destroyAllWindows()
