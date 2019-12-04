
# **Model of Recognition and Classification of product with the purpose of automatic payment for Distributors**



## Table of Contents.
* [About](#about)
* [Architecture](#architecture)
* [Result](#result)
* Reference


## About
created a prototype system based on pre-trained RCNN/CNN models that is able to recognize products and classify what is the product in an image or realtime cam.

The prototype is designed to be implemented in the large scale distributor or retail industry for automatic payment:
* Reduce payment process

> No lift up product, no find barcode, no scan barcode using reader.
> just show purchasing product, end.

* low cost about man power

> Before prototype, always one counter one man. Now, staffs come only when customer who first time this type of purchase called.
> stay will be unnecessary.

* connectivity to mobile pay

> mobile is not simple calling machine. this can be identification, purchase measure.
> if prototype connected mobile pay, would be more convenient, fast and secure.



## Architecture
Stages | Preprocessing | Object Detection | Product Classification
-------|---------------|------------------|------------------------
**ML Models** | | SSD, MaskRCNN | Inception, MLP
**Libraries** | OpenCV | Caffe, Keras | Tensorflow, Keras
**Language** | Python | Python | Python


## Result
![grab-landing-page](https://github.com/rib-president/Hide-on-bush/blob/master/sample/output.gif)

Covered class of RCNN is proportionally small than CNN.
Collaborate object detection and classification to handled more big dataset, to made more accurate result.








