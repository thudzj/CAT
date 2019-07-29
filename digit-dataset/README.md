# Cluster Alignment with a Teacher for Unsupervised Domain Adaptation
## Try CAT on digits adaptation tasks:
```
python train.py --source svhn --target mnist
```
## Try CAT+RevGrad on digits adaptation tasks:
```
python train.py --source svhn --target mnist --revgrad 1
```
## Try CAT+rRevGrad on digits adaptation tasks:
```
python train_r.py --source svhn --target mnist
```
## Try CAT on imbalanced digits adaptation tasks:
```
python train.py --source svhn --target mnist --unbalance 10
```

The datasets would be dowloaded automatically.

