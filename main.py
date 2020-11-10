import cv2
import numpy as np
import sys
import math
import time
from skimage.metrics import structural_similarity as ssim

FNAME = 'digits1106.npz'

def machineLearning():
    img = cv2.imread('images/digits.png')
    img = img[100:400,:]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cells = [np.hsplit(row,100) for row in np.vsplit(gray,15)]

    x = np.array(cells)
    train = x[:,:].reshape(-1,400).astype(np.float32)

    k = np.arange(3)+1
    train_labels = np.repeat(k,500)[:,np.newaxis]

    np.savez(FNAME,train=train,train_labels = train_labels)

def resize20(frame):
    if sys.argv[1] == 'train':
        img = cv2.imread(frame)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        grayResize = cv2.resize(gray, (20, 20))
        ret, thresh = cv2.threshold(grayResize, 125, 255, cv2.THRESH_BINARY_INV)

    elif sys.argv[1] == 'test':
        if frame.shape[0] == 1080:
            imcp = frame[81:130, 712:743]
        else:
            imcp = frame[35:70,350:380]

        gray = cv2.cvtColor(imcp, cv2.COLOR_BGR2GRAY)
        grayResize = cv2.resize(gray,(20,20))
        thresh = cv2.adaptiveThreshold(grayResize, 255, 1, 1, 11, 2)

    return thresh.reshape(-1,400).astype(np.float32)

def loadTrainData(fname):
    with np.load(fname) as data:
        train = data['train']
        train_labels = data['train_labels']

    return train, train_labels

def checkDigit(test, train, train_labels):
    knn = cv2.ml.KNearest_create()
    knn.train(train, cv2.ml.ROW_SAMPLE, train_labels)

    ret, result, neighbours, dist = knn.findNearest(test, k=5)

    return result

if __name__ == '__main__':
    if len(sys.argv) == 1:
        print('option : train or test')
        exit(1)
    elif sys.argv[1] == 'train':
        machineLearning()
    elif sys.argv[1] == 'test':
        fname = sys.argv[2]
        cam = cv2.VideoCapture(fname)
        train, train_labels = loadTrainData(FNAME)
        saveNpz = False
        _, input = cam.read()
        test = resize20(input)
        result = checkDigit(test, train, train_labels)

        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(sys.argv[3], fourcc, 30.0, (input.shape[1],input.shape[0]))

        trans_start = False
        trans_end = False
        fn = 0
        while (cam.isOpened()):
            start = time.time()
            prv_frame = input.copy()

            if str(type(result)) == "<class 'int'>":
                before = int(result[0][0])
            else:
                before = result

            if input.shape[0] == 1080:
                prcrop_input = prv_frame[0:350, 0:135]
            else:
                prcrop_input = prv_frame[0:75, 0:300]
            ret, input = cam.read()

            if ret:
                current_frame = input.copy()
                if input.shape[0] == 1080:
                    crcrop_input = current_frame[0:350, 0:135]
                else:
                    crcrop_input = current_frame[0:75, 0:300]
                
                pcssim = ssim(prcrop_input, crcrop_input, multichannel=True)

                current_frame = input.copy()
                if input.shape[0] == 1080:
                    cv2.rectangle(current_frame, (0, 0), (350, 135), (0, 0, 255), 2)
                else:
                    cv2.rectangle(current_frame, (0, 0), (1080, 75), (0, 0, 255), 2)
                test = resize20(current_frame)
                if np.count_nonzero(test) < 30:
                    result = "None"
                    cv2.putText(current_frame, "Preset: " + result, (40, 200), 0,
                                3, (0, 255, 0), 2)
                    current = result
                else:
                    result = checkDigit(test, train, train_labels)
                    cv2.putText(current_frame, "Preset: " + str(int(result[0][0])), (40, 200), 0,
                                3, (0, 255, 0), 2)
                    current = int(result[0][0])
                cv2.putText(current_frame, "ssim: " + str(round(pcssim,2)), (700, 200), 0,
                            2, (0, 255, 0), 2)

                cv2.imshow("frame", current_frame)

                k = cv2.waitKey(0)

                if k > 47 and k < 58:
                    print("change")
                    saveNpz = True
                    train = np.append(train, test, axis=0)
                    newLabel = np.array(int(chr(k))).reshape(-1, 1)
                    train_labels = np.append(train_labels, newLabel, axis=0)
                    if saveNpz:
                        np.savez(FNAME, train=train, train_labels=train_labels)
                elif k == 13:
                    print("pass for None sign")
                    pass
                out.write(current_frame)
                fn += 1
            else:
                break

        out.release()
        cam.release()
        cv2.destroyAllWindows()
        print("done")

    else:
        print ('unknow option')
