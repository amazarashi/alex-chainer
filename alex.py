import chainer
import chainer.functions as F
import chainer.links as L
import skimage.io as io
import numpy as np
from chainer import utils

class Alex(chainer.Chain):

    def __init__(self,category_num=10):
        initializer = chainer.initializers.HeNormal()
        super(Alex, self).__init__(
            conv1=L.Convolution2D(3,  96, 11, stride=4,initialW=initializer),
            bn1 = L.BatchNormalization(96),
            conv2=L.Convolution2D(96, 256,  5, pad=2,initialW=initializer),
            bn2 = L.BatchNormalization(256),
            conv3=L.Convolution2D(256, 384,  3, pad=1,initialW=initializer),
            bn3 = L.BatchNormalization(384),
            conv4=L.Convolution2D(384, 384,  3, pad=1,initialW=initializer),
            bn4 = L.BatchNormalization(384),
            conv5=L.Convolution2D(384, 256,  3, pad=1,initialW=initializer),
            bn5 = L.BatchNormalization(256),
            fc6=L.Linear(9216, 4096,initialW=initializer),
            fc7=L.Linear(4096, 4096,initialW=initializer),
            fc8=L.Linear(4096, 1024),
            fc9=L.Linear(1024, category_num),
        )

    def __call__(self,x,train=True):
        initializer = chainer.initializers.HeNormal()
        #x = chainer.Variable(x)
        h = F.relu(self.conv1(x))
        h = self.bn1(h, test=not train)
        h = F.local_response_normalization(h)
        h = F.max_pooling_2d(h, 3, stride=2)

        h = F.relu(self.conv2(h))
        h = self.bn2(h, test=not train)
        h = F.local_response_normalization(h)
        h = F.max_pooling_2d(h, 3, stride=2)

        h = F.relu(self.conv3(h))
        h = self.bn3(h, test=not train)
        h = F.relu(self.conv4(h))
        h = self.bn4(h, test=not train)
        h = F.relu(self.conv5(h))
        h = self.bn5(h, test=not train)
        h = F.max_pooling_2d(h, 3, stride=2)

        h = F.dropout(F.relu(self.fc6(h)), train=train)
        h = F.dropout(F.relu(self.fc7(h)), train=train)
        h = F.dropout(F.relu(self.fc8(h)), train=train)
        h = self.fc9(h)
        return h

    def calc_loss(self,y,t):
        loss = F.softmax_cross_entropy(y,t)
        return loss

    def accuracy_of_each_category(self,y,t):
        y.to_cpu()
        t.to_cpu()
        categories = set(t.data)
        accuracy = {}
        for category in categories:
            supervise_indices = np.where(t.data==category)[0]
            predict_result_of_category = np.argmax(y.data[supervise_indices],axis=1)
            countup = len(np.where(predict_result_of_category==category)[0])
            accuracy[category] = countup
        return accuracy

#if __name__ == "__main__":
    # imgpath = "/Users/suguru/Desktop/test.jpg"
    # img = io.imread(imgpath)
    # img = np.asarray(img).transpose(2,0,1).astype(np.float32)/255.
    # img = img[np.newaxis]
    # ex = model(img)
    # print(ex)
