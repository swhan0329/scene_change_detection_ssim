# scene_change_detection_ssim
 Detecting scene changes using the SSIM metric

# Prior knowledge
## SSIM Metric
The structural similarity index measure (SSIM) is a method for predicting the perceived quality of digital television and cinematic pictures, as well as other kinds of digital images and videos. SSIM is used for measuring the similarity between two images. The SSIM index is a full reference metric; in other words, the measurement or prediction of image quality is based on an initial uncompressed or distortion-free image as reference.

Reference: https://en.wikipedia.org/wiki/Structural_similarity

## Detection Number using the KNN algorithm

Reference: https://opencv-python.readthedocs.io/en/latest/doc/29.matchDigits/matchDigits.html

I need to recognize number 1, 2, 3 for preset information. So I trained only 1, 2, 3.

# Method
1. Crop the specific area. I use red box area in the below image.

![image][red_box]

[red_box]: https://github.com/swhan0329/scene_change_detection_ssim/blob/main/red_box.PNG?raw=true "Red box in the image"

2. Compare the two scenes using the SSIM metric. You can get the SSIM metric using skimage libarary.

```python
from skimage.metrics import structural_similarity as ssim

pcssim = ssim(prcrop_input, crcrop_input, multichannel=True)
```

# How to run this code
1. You should train this code firstly using below an image and code.

![image][train_image]

[train_image]: https://opencv-python.readthedocs.io/en/latest/_images/image015.png "Image for the training"

```bash
python main.py train
```

2. This step is called test, but you can train your trained weight or see the result.

```bash
python main.py test [input_video] [output_video]
```

ex)
```bash
python main.py test CH4_input.avi CH4_output.mp4
```

2-1. [train] Trained weight using 1 step is not accurate. So you should overfit your train data. When you see the frame of input video, you should correct answers compared to predicted answer. Just press number keys.

2-2. [test] If you want to see the result, comment below codes and just run the code.

```python
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
```

# Result
[![Video Label](http://img.youtube.com/vi/xlSjDU0XAvw/0.jpg)](https://youtu.be/xlSjDU0XAvw?t=0s)
