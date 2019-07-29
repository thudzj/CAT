import numpy as np
from sklearn.cluster import KMeans

num_classes = 10
dirr = "./"
cat = np.load(dirr+"features_cat.npz")
rev = np.load(dirr+"features_rev.npz")
mstn = np.load(dirr+"features_mstn.npz")

cat_pred = KMeans(n_clusters=num_classes, n_jobs=-1).fit_predict(np.concatenate([cat['x1'],cat['x2']] , 0))
cat_label = np.concatenate([cat['y1'],cat['y2']] , 0)
components = {}
labels = {}
correct = 0
summ = 0
for i in range(num_classes):
    components[i] = np.nonzero(cat_pred == i)[0]
    #print(components[i].shape)
    tmp = []
    for j in range(num_classes):
        tmp.append((cat_label[components[i]] == j).sum())
    #print(tmp)
    labels[i] = np.argmax(np.array(tmp))
    correct += np.max(np.array(tmp))
    summ += np.sum(np.array(tmp))
    #print(labels[i])
print(float(correct) / summ)

cat_pred = KMeans(n_clusters=num_classes, n_jobs=-1).fit_predict(np.concatenate([rev['x1'],rev['x2']] , 0))
cat_label = np.concatenate([rev['y1'],rev['y2']] , 0)
components = {}
labels = {}
correct = 0
summ = 0
for i in range(num_classes):
    components[i] = np.nonzero(cat_pred == i)[0]
    #print(components[i].shape)
    tmp = []
    for j in range(num_classes):
        tmp.append((cat_label[components[i]] == j).sum())
    #print(tmp)
    labels[i] = np.argmax(np.array(tmp))
    correct += np.max(np.array(tmp))
    summ += np.sum(np.array(tmp))
    #print(labels[i])
print(float(correct) / summ)


cat_pred = KMeans(n_clusters=num_classes, n_jobs=-1).fit_predict(np.concatenate([mstn['x1'],mstn['x2']] , 0))
cat_label = np.concatenate([mstn['y1'],mstn['y2']] , 0)
components = {}
labels = {}
correct = 0
summ = 0
for i in range(num_classes):
    components[i] = np.nonzero(cat_pred == i)[0]
    #print(components[i].shape)
    tmp = []
    for j in range(num_classes):
        tmp.append((cat_label[components[i]] == j).sum())
    #print(tmp)
    labels[i] = np.argmax(np.array(tmp))
    correct += np.max(np.array(tmp))
    summ += np.sum(np.array(tmp))
    #print(labels[i])
print(float(correct) / summ)
