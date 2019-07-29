import numpy
import os
import sys
import util
from urlparse import urljoin
import gzip
import struct
import operator
import numpy as np
#from preprocessing import preprocessing
def get_one_hot(targets, nb_classes):
    return np.eye(nb_classes)[np.array(targets).reshape(-1)]
class USPS:
        base_url = 'http://statweb.stanford.edu/~tibs/ElemStatLearn/datasets/'

        data_files = {
        'train': 'zip.train.gz',
        'test': 'zip.test.gz'
        }
        def __init__(self,path=None,shuffle=True,output_size=[16,16],output_channel=1,split='train',select=[], unbalance=1.):
                self.image_shape=(16,16,1)
                self.label_shape=()     
                self.path=path
                self.shuffle=shuffle
                self.output_size=output_size
                self.output_channel=output_channel
                self.split=split
                self.select=select
                self.unbalance=unbalance
                self.num_classes = 10 if unbalance == 1. else 2
                self.download()
                self.pointer=0
                self.load_dataset()
        def download(self):
                data_dir = self.path
                if not os.path.exists(data_dir):
                        os.mkdir(data_dir)
                for filename in self.data_files.values():
                        path = self.path+'/'+filename
                        if not os.path.exists(path):
                                url = urljoin(self.base_url, filename)
                                util.maybe_download(url, path)
        def shuffle_data(self):
                images = self.images[:]
                labels = self.labels[:]
                self.images = []
                self.labels = []

                idx = np.random.permutation(len(labels))
                for i in idx:
                        self.images.append(images[i])
                        self.labels.append(labels[i])
        def _read_datafile(self, path):
                """Read the proprietary USPS digits data file."""
                labels, images = [], []
                with gzip.GzipFile(path) as f:
                    for line in f:
                        vals = line.strip().split()
                        labels.append(float(vals[0]))
                        images.append([float(val) for val in vals[1:]])
                labels = np.array(labels, dtype=np.int32)
                labels[labels == 10] = 0  # fix weird 0 labels
                images = np.array(images, dtype=np.float32).reshape(-1, 16, 16, 1)
                images = (images + 1) / 2
                return images, labels

        def load_dataset(self):
                abspaths = {name: self.path+'/'+path
                        for name, path in self.data_files.items()}
                if self.split=='train':
                        self.images, self.labels = self._read_datafile(abspaths['train'])
                elif self.split=='test':
                        self.images, self.labels = self._read_datafile(abspaths['test'])
                if self.unbalance == 1.:
                        if len(self.select)!=0:
                                self.images=self.images[self.select]
                                self.labels=self.labels[self.select]
                else:
                        if self.unbalance > 1.:
                                base = 1000
                        else:
                                base = 100
                        img1 = self.images[self.labels == 0][:base]
                        lbl1 = self.labels[self.labels == 0][:base]
                        img2 = self.images[self.labels == 1][:int(base/self.unbalance)]
                        lbl2 = self.labels[self.labels == 1][:int(base/self.unbalance)]
                        self.images = np.concatenate([img1, img2], 0)
                        self.labels = np.concatenate([lbl1, lbl2], 0)


        
        def reset_pointer(self):
                self.pointer=0
                if self.shuffle:
                        self.shuffle_data()     

        def class_next_batch(self,num_per_class):
                batch_size=10*num_per_class
                classpaths=[]
                ids=[]
                for i in xrange(10):
                        classpaths.append([])
                for j in xrange(len(self.labels)):
                        label=self.labels[j]
                        classpaths[label].append(j)
                for i in xrange(10):
                        ids+=np.random.choice(classpaths[i],size=num_per_class,replace=False).tolist()
                selfimages=np.array(self.images)
                selflabels=np.array(self.labels)
                return np.array(selfimages[ids]),get_one_hot(selflabels[ids],self.num_classes)

        def next_batch(self,batch_size):
                if self.pointer+batch_size>len(self.labels):
                        self.reset_pointer()
                images=self.images[self.pointer:(self.pointer+batch_size)]
                labels=self.labels[self.pointer:(self.pointer+batch_size)]
                self.pointer+=batch_size
                return np.array(images),get_one_hot(labels,self.num_classes)  
        

def main():
        mnist=USPS(path='data/usps')
        print(mnist.images.max(), mnist.images.min(), mnist.images.shape)
        a,b=mnist.next_batch(1)
        

if __name__=='__main__':
        main()
