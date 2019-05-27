# pedestrian_recognition
A simple human recognition api for re-ID usage, power by paper [In Defense of the Triplet Loss for Person Re-Identification](https://arxiv.org/abs/1703.07737) and [MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications]( https://arxiv.org/abs/1704.04861)


## Testing Environment
### Operating system
1. MacOS Sierra 
2. Ubuntu 16.04

### Python package (Python 3.5 or Python3.6)
1. Tensorflow 1.8 
2. opencv 3.3 (Need opencv dnn library)
3. Numpy

## Workflow
1. Download model from Google drive https://drive.google.com/file/d/1-2KHQms_RHYIrHwFQg-Vln-NEyhjj5_p/view?usp=sharing
2. Unzip ```unzip model.zip```
3. Use opencv dnn module and use caffemodel to detection human in an image.
4. Crop and resize all human(pedestrian) and resize to 256x128 images.
5. Put image to resnet-50 human feature embedding extractor and get a 128-D feature array.
6. Compare two human by using euclidean distance, the distance means the similarity of two image.

## Example code
```
img1 = cv2.imread('test/test1.png')[:,:,::-1]
img1_location = api.human_locations(img1)
img_1_human = api.crop_human(img1, img1_location)
human_1_1 = img_1_human[0]
human_1_1_vector = api.human_vector(human_1_1)
# Do another people, and compare
```

## Add Mobilenet backbone support
Thanks to the original repo, I trained a mobilenet backbone model which can accerlerate the speed of human embedding. You can check the time difference between mobilenet and resnet-50

Also, attached is the mobilenet backbone pretrained model that I trained.
Here is the google drive link:
https://drive.google.com/file/d/1JoJJ-rIrqXNrzrx12Ih4zFk09SYsKINC/view?usp=sharing

And the evaluation score of the model is:
```
mAP: 66.28% | top-1: 83.11% top-2: 88.42% | top-5: 93.79% | top-10: 95.90%
```
![GitHub Logo](https://github.com/cftang0827/human_recognition/blob/mobilenet/mobilenet_train_result.png?raw=true)


Please use mobilenet branch and download the pretrained model from the link and replace original resnet model

## Acknowledgement and reference
1. https://github.com/VisualComputingInstitute/triplet-reid
2. https://github.com/chuanqi305/MobileNet-SSD
3. https://github.com/opencv/opencv/tree/master/samples/dnn


```
@article{HermansBeyer2017Arxiv,
  title       = {{In Defense of the Triplet Loss for Person Re-Identification}},
  author      = {Hermans*, Alexander and Beyer*, Lucas and Leibe, Bastian},
  journal     = {arXiv preprint arXiv:1703.07737},
  year        = {2017}
}
```
