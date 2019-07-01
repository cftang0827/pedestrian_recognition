import tensorflow as tf 
import numpy as np 
import cv2 
import nets.resnet_v1_50 as model
import heads.fc1024 as head

# Tensorflow human re-ID feature descriptor model
tf.Graph().as_default()
sess = tf.Session()
images = tf.zeros([1, 256, 128, 3], dtype=tf.float32)
endpoints, body_prefix = model.endpoints(images, is_training=False)
with tf.name_scope('head'):
    endpoints = head.head(endpoints, 128, is_training=False)
tf.train.Saver().restore(sess, 'model/checkpoint-25000')


# caffe mobilenet model
net = cv2.dnn.readNetFromCaffe(
    'model/deploy.prototxt', 'model/MobileNetSSD_deploy.caffemodel')
classNames = {0: 'background', 15: 'person'}

def human_vector(img):
    resize_img = cv2.resize(img, (128,256))
    resize_img = np.expand_dims(resize_img, axis=0)
    emb = sess.run(endpoints['emb'], feed_dict={images: resize_img})
    return emb

def human_distance(enc1, enc2):
    return np.sqrt(np.sum(np.square(enc1 - enc2)))

def crop_human(frame, locations):
    human_image = []
    for loc in locations:
        leftBottom, rightTop = loc
        sub_frame = frame[leftBottom[1]:rightTop[1], leftBottom[0]:rightTop[0], :]  # 3 channel image
        human_image.append(sub_frame)
    return human_image

def human_locations(frame, thr=0.5):
    frame_resized = cv2.resize(frame, (300,300))
    blob = cv2.dnn.blobFromImage(
        frame_resized, 0.007843, (300, 300), (127.5, 127.5, 127.5), False)
    net.setInput(blob)
    #Prediction of network
    detections = net.forward()
    #Size of frame resize (300x300)
    cols = frame_resized.shape[1] 
    rows = frame_resized.shape[0]
    output = []

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2] #Confidence of prediction 
        if confidence > thr: # Filter prediction 
            class_id = int(detections[0, 0, i, 1]) # Class label

            # Object location 
            xLeftBottom = int(detections[0, 0, i, 3] * cols) 
            yLeftBottom = int(detections[0, 0, i, 4] * rows)
            xRightTop   = int(detections[0, 0, i, 5] * cols)
            yRightTop   = int(detections[0, 0, i, 6] * rows)
            
            # Factor for scale to original size of frame
            heightFactor = frame.shape[0]/300.0  
            widthFactor = frame.shape[1]/300.0 
            # Scale object detection to frame
            xLeftBottom = max(0, int(widthFactor * xLeftBottom)) 
            yLeftBottom = max(0, int(heightFactor * yLeftBottom))
            xRightTop   = max(0, int(widthFactor * xRightTop))
            yRightTop   = max(0, int(heightFactor * yRightTop))
            if class_id in classNames:
                output.append([(xLeftBottom, yLeftBottom), (xRightTop, yRightTop)])

    return output
