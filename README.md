<img src="https://jhonykaesemodel.com/img/headers/overview_image2mesh.png" width="800">

# Image2Mesh: A learning framework for single image 3D reconstruction
Here you'll find the codes for the paper **[Image2Mesh: A learning framework for single image 3D reconstruction](https://jhonykaesemodel.com/publication/image2mesh/)**.

## Datasets Used
In this work we used the **[ShapeNetCore.v1](https://www.shapenet.org/)** and **[PASCAL3D+\_release1.1](http://cvgl.stanford.edu/projects/pascal3d.html)** datasets.

## Getting Started
Running a demo for the aeroplane category:
1. Clone the repos:
```
git clone https://github.com/jhonykaesemodel/compact_3D_reconstruction.git
git clone https://github.com/jhonykaesemodel/image2mesh
```
2. Download the ShapeNetCore 3D model IDs used in this work and its manually annotated 3D anchors **[here](https://www.dropbox.com/s/f2895gpuclqvvpt/data_demo.zip?dl=0)**
3. Follow the instructions in the `get_started.m` file and have fun :)


**Note:** The file `\image2mesh\shapenet\ShapeNetCore_IDs_8_classes.txt` contains all the 3D model IDs for the 8 classes used in this work.


## Reference
If you find this code useful in your research, please cite the [**paper**](https://arxiv.org/abs/1711.10669.pdf):
```
@article{pontes2017image2mesh,
  title={Image2Mesh: A Learning Framework for Single Image 3D Reconstruction},
  author={Jhony K. Pontes and Chen Kong and Sridha Sridharan and Simon Lucey and Anders Eriksson and Clinton Fookes},
  journal={arXiv:1711.10669},
  year={2017}
}
```
