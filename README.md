# BING
[BING: Binarized Normed Gradients for Objectness Estimation at 300fps](http://mmcheng.net/bing/)

This repository provides C++ code and matlab wrappers for both windows and linux.

### Installation for the datasets

This package provides datasets for VOC2007 and COCO.

**1. Installation for VOC2007**

 - For VOC2007, extract datasets/VOC2007/Annotations.tar.gz and datasets/VOC2007/ImageSets.tar.gz into VOC2007 folder.
 - If you don't want to train the model of BING by yourself, datasets/VOC2007/Results.tar.gz should be extracted into VOC2007 folder, too.
 - JPEGImages folder from VOC2007 dataset is not included due to its size.
 
So, the VOC2007 folder should have this basic structure:

```
$VOC2007/
$VOC2007/Annotations/
$VOC2007/ImageSets/
$VOC2007/JPEGImages/
$VOC2007/Results/
```

**2. Installation for COCO**

The installation for COCO is similar to VOC2007. 

 - Since COCO only has train set and val set, we regard val set as test set here.
 - The names of images in COCO are like COCO_val2014_000000000042.jpg. However, we only use the last six number of their names for convenience, like 000042.jpg. And you should put images of train set and val set into JPEGImages folder together. 

Make the COCO folder like:

```
$COCO/
$COCO/Annotations/
$COCO/ImageSets/
$COCO/JPEGImages/
$COCO/Results/
```

### Installation for the software

This package provides code for Linux and Windows. 

**1. Installation on Linux**

Tested on Ubuntu 15.04 with OpenCV 3.0.

To build:
```
mkdir build
cd build
cmake ..
make
```

To run:
```
./BING/BING /path/to/data/ (e.g. /datasets/VOC2007/)
```

**2. Installation on Windows**

Tested on Windows 10 with Visual Studio 2013, Opencv 3.0.

**Notes:**

 - You can change the path of dataset in line 24, Main.cpp.
 - Please run the code under Release, x64 mode.
 
**3. Installation for matlab wrappers**

First, configure environment as described above. Then, build.m contains some commands to build the mex function; mexBING is the wrapper of BING algorithm; and runDataset.m use mexBING to run the dataset.

