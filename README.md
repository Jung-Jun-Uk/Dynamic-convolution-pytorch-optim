# Dynamic Convolution (training optimization)

Paper: [Dynamic Convolution: Attention over Convolution Kernels](https://arxiv.org/pdf/1912.03458.pdf)


Implementation with reference to https://github.com/kaijieshi7/Dynamic-convolution-Pytorch 

The training time is __about 7 times faster__ on the cifar10 dataset.

### Check
```python
python dyconv2d.py
```

### Training
```python
python train.py 
    --device 'cuda device, i.e. 0 or 0,1,2,3 or cpu'
    --training_optim #training more faster
```

### Inference
just call model.inference_mode()
```python
model = DyMobileNetV2(num_classes=opt.num_classes, input_size=32, width_mult=1.)
model.inference_mode()
```