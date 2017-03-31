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

[ORIGINAL (for ImageNet)]<br/>
fc6=L.Linear(9216, 4096,initialW=initializer),<br/>
fc7=L.Linear(4096, 4096,initialW=initializer),<br/>
fc8=L.Linear(4096, 1024,initialW=initializer),<br/>
fc9=L.Linear(1024, 10),<br/>

# How to run

git clone git@github.com:amazarashi/alex-chainer.git

cd ./squeeze-chainer

python main.py -g 1

# Inspection

## dataset

 - Cifar10 [link](https://www.cs.toronto.edu/~kriz/cifar.html)

## Result
In this inspection, I tried 2-types of model, original AlexNet model and original AlexNet model with batch normalization.
the result on each model is as followed..

### original AlexNet Model with batch normalization

#### Accuracy
![withBN_accuracy](https://github.com/amazarashi/alex-chainer/blob/feature/bn/result/model_custom/accuracy.png?raw=true "")

#### Loss
![withBN_loss](https://github.com/amazarashi/alex-chainer/blob/feature/bn/result/model_custom/loss.png?raw=true "")


### original AlexNet Model

#### Accuracy
![original_accuracy](https://github.com/amazarashi/alex-chainer/blob/feature/bn/result/model_original/accuracy.png?raw=true "")

#### Loss
![original_loss](https://github.com/amazarashi/alex-chainer/blob/feature/bn/result/model_original/loss.png?raw=true "")

