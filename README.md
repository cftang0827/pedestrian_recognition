# human_recognition
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
1. Download model from Google drive https://drive.google.com/open?id=1kql_aaM-rnMISOzl1Y2Fq5XHpxad8gMT
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
