import cv2
import numpy as np
import pytesseract
import time


from flask import Flask, url_for,  send_from_directory
from flask import render_template
import os
import argparse
import sys

app = Flask(__name__)


@app.route('/')
def my_form():
    return render_template("hey.html") # this should be the name of your html file

@app.route('/video_feed')
def video_feed():
    # Load the YOLO model
    model_weight = "safa-yalla/removision/blob/main/yolov4-tiny.weights"
    model_cfg = "safa-yalla/removision/blob/main/yolov4_tiny.cfg"
    net = cv2.dnn.readNet(model_weight, model_cfg)
    classes = []
    with open("safa-yalla/removision/blob/main/coco.names",
              "r") as f:
        classes = [line.strip() for line in f.readlines()]
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
    colors = np.random.uniform(0, 255, size=(len(classes), 3))

    # Load webcam
    cap = cv2.VideoCapture(0)
    font = cv2.FONT_HERSHEY_SIMPLEX
    starting_time = time.time()
    frame_id = 0

    while True:
        # Read webcam
        _, frame = cap.read()
        frame_id += 1
        height, width, channels = frame.shape

        # Detecting objects
        cv2.dnn.blobFromImage(frame, 1, (64, 64), -127, swapRB=False, crop=False)
        blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        net.setInput(blob)
        outs = net.forward(output_layers)

        # Visualising data
        class_ids = []
        confidences = []
        boxes = []
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.1:
                    # Object detected
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)

                    # Rectangle coordinates
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)

                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.8, 0.3)

        for i in range(len(boxes)):
            if i in indexes:
                x, y, w, h = boxes[i]
                label = str(classes[class_ids[i]])
                print(label)
                confidence = confidences[i]
                color = colors[class_ids[i]]
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                cv2.putText(frame, label + " " + str(round(confidence, 2)), (x, y + 30), font, 3, color, 3)

        elapsed_time = time.time() - starting_time
        fps = frame_id / elapsed_time
        cv2.putText(frame, "FPS: " + str(round(fps, 2)), (40, 670), font, .7, (0, 255, 255), 1)
        cv2.putText(frame, "press [esc] to exit", (40, 690), font, .45, (0, 255, 255), 1)

        cv2.imshow("Image", frame)
        key = cv2.waitKey(1)
        if key == 27:
            print("[button pressed] ///// [esc].")
            print("[feedback] ///// Videocapturing succesfully stopped")
            break

    cap.release()
    cv2.destroyAllWindows()
    return render_template("hey.html")

@app.route('/removal1')
def removal1():
    pytesseract.pytesseract.tesseract_cmd = r"safa-yalla/removision/blob/main/tesseract.exe"
    file_path = "safa-yalla/removision/blob/main/static/text_img.jpg"
    # Load image, grayscale, Otsu's threshold
    image = cv2.imread(file_path)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower = np.array([0, 0, 0])
    upper = np.array([100, 175, 110])
    mask = cv2.inRange(hsv, lower, upper)

    # Invert image and OCR
    invert = 255 - mask
    data = pytesseract.image_to_string(invert, lang='eng', config='--psm 6')
    print(data)

    cv2.imshow('mask', mask)
    cv2.imshow('invert', invert)
    cv2.waitKey()
    cv2.destroyAllWindows()
    return render_template("hey.html")

@app.route('/denoise')
def denoise():
    file_path = "safa-yalla/removision/blob/main/static/DiscoveryMuseum_NoiseAdded.jpg"
    img = cv2.imread(file_path)
    denoise_1 = cv2.fastNlMeansDenoisingColored(img, None, 3, 3, 7, 21)
    denoise_2 = cv2.fastNlMeansDenoisingColored(img, None, 5, 5, 7, 21)
    denoise_3 = cv2.fastNlMeansDenoisingColored(img, None, 15, 15, 7, 21)
    cv2.imshow('Original Image', img)
    cv2.imshow('image_1', denoise_1)
    cv2.imshow('image_2', denoise_2)
    cv2.imshow('image_3', denoise_3)
    cv2.waitKey()
    cv2.destroyAllWindows()
    return render_template("hey.html")


@app.route('/removebg')
def removebg():
    # == Parameters =======================================================================
    BLUR = 21
    CANNY_THRESH_1 = 10
    CANNY_THRESH_2 = 200
    MASK_DILATE_ITER = 10
    MASK_ERODE_ITER = 10
    MASK_COLOR = (0.0, 0.0, 0.0)  # In BGR format

    # == Processing =======================================================================

    # -- Read image -----------------------------------------------------------------------
    file_path = "safa-yalla/removision/blob/main/static/male-caucasian-person-notebook-looking-260nw-1203432"
    img = cv2.imread(file_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # -- Edge detection -------------------------------------------------------------------
    edges = cv2.Canny(gray, CANNY_THRESH_1, CANNY_THRESH_2)
    edges = cv2.dilate(edges, None)
    edges = cv2.erode(edges, None)

    # -- Find contours in edges, sort by area ---------------------------------------------
    contour_info = []
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    # Previously, for a previous version of cv2, this line was:
    #  contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    # Thanks to notes from commenters, I've updated the code but left this note
    for c in contours:
        contour_info.append((
            c,
            cv2.isContourConvex(c),
            cv2.contourArea(c),
        ))
    contour_info = sorted(contour_info, key=lambda c: c[2], reverse=True)
    max_contour = contour_info[0]

    # -- Create empty mask, draw filled polygon on it corresponding to largest contour ----
    # Mask is black, polygon is white
    mask = np.zeros(edges.shape)
    cv2.fillConvexPoly(mask, max_contour[0], (255))

    # -- Smooth mask, then blur it --------------------------------------------------------
    mask = cv2.dilate(mask, None, iterations=MASK_DILATE_ITER)
    mask = cv2.erode(mask, None, iterations=MASK_ERODE_ITER)
    mask = cv2.GaussianBlur(mask, (BLUR, BLUR), 0)
    mask_stack = np.dstack([mask] * 3)  # Create 3-channel alpha mask

    # -- Blend masked img into MASK_COLOR background --------------------------------------
    mask_stack = mask_stack.astype('float32') / 255.0  # Use float matrices,
    img = img.astype('float32') / 255.0  # for easy blending

    masked = (mask_stack * img) + ((1 - mask_stack) * MASK_COLOR)  # Blend
    masked = (masked * 255).astype('uint8')  # Convert back to 8-bit

    cv2.imshow('img', masked)  # Display
    cv2.waitKey()
    return render_template("hey.html")
port = int(os.environ.get('PORT', 5000))


#cv2.imwrite('C:/Temp/person-masked.jpg', masked)           # Save
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=port, debug=True)

