## FaceSwap

Swapping two faces from two different frames using classical and deep learning computer vision approaches.

---

## Contributors

1) [Aditya Jadhav](https://github.com/iamjadhav)
Graduate Student of M.Eng Robotics at University of Maryland. 
2) [Abhishek Nalawade](https://github.com/abhishek-nalawade)
Graduate Student of M.Eng Robotics at University of Maryland.

## Overview

• An end-to-end pipeline to swap faces in videos and images using warping methods Delaunay Triangulation and Th Landin Plate Spline. 
  Obtaining Face landmarks by dlib and using getTraingleList() and subdiv2D() to implement Delaunay method to swap.
• Thin Plate Spline model parameters calculation to map Source to Destination face, then warping all Source pixels to Destination face pixels.


## Technology Used


* Ubuntu 18.04 LTS
* Modern C++ Programming Language
* OpenCV Library

## License 

```
MIT License

Copyright (c) 2021 Abhishek Nalawade, Aditya Jadhav

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE 
SOFTWARE.
```

## Set of Assumptions 

- Humans faces are present in all image and videos.
- Videos have two human faces and no more (preferrably).

## Known Issues/Bugs 

- Not a Robust Implementation for head rotations and tilts.

## Dependencies

- Install OpenCV 3.4.4 and other dependencies using this link. Refer [OpenCV](https://learnopencv.com/install-opencv-3-4-4-on-ubuntu-18-04/)

## How to build

```
git clone --recursive https://github.com/iamjadhav/faceswap.git
cd faceswap
```

Mode 1 - Image to Video Frame (change method to tps or prnet)
```
python Wrapper.py --method delaunay
```
Mode 2 -  Video face swap
```
python Wrapper.py --method delaunay --mode 2 --dst_frame TestData/Test10.mp4
```

## Links

Demo --> [Link](https://iamjadhav.myportfolio.com/computer-vision)
