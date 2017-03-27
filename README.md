# ※ [STILL DEVELOPIING !!]

# About
AlexNet by chainer

# Paper

[ImageNet Classification with Deep Convolutional
Neural Networks](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf)
# Model

AlexNet (Customized)

In this inspection, image(128x128x2) which was scaled and random croped against original cifar10-image(32x32x3) is used.
So, in this implemention, customizing parameter on fc-layer. 

In detail

[ORIGINAL (for ImageNet)]<br/>
fc6=L.Linear(9216, 4096,initialW=initializer),<br/>
fc7=L.Linear(4096, 4096,initialW=initializer),<br/>
fc8=L.Linear(4096, 1000),<br/>

↓<br/>

[CUSTOMiZED (for Cifar10)]<br/>
fc6=L.Linear(2304, 1024,initialW=initializer),<br/>
fc7=L.Linear(1024, 1024,initialW=initializer),<br/>
fc8=L.Linear(1024, 10),<br/>

# How to run

git clone git@github.com:amazarashi/alex-chainer.git

cd ./squeeze-chainer

python main.py -g 1

# Inspection

#### dataset

 - Cifar10 [link](https://www.cs.toronto.edu/~kriz/cifar.html)

#### Result

Coming Soon..
