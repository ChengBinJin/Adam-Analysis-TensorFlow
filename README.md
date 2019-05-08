# Adam-Analysis-TensorFlow
This repository is an evaluation of Kingma's ["ADAM: A Method for Stochastic Optimization"](https://arxiv.org/pdf/1412.6980.pdf) adam optimizer and others.  

<p align="center">
<img src="https://user-images.githubusercontent.com/37034031/57378858-98b32100-71e0-11e9-92c9-62c20e9a167e.png" width=700>
</p>

## Requirements
- tensorflow 1.13.1
- pickleshare 0.7.4
- numpy 1.15.2
- unicodecsv 0.14.1
- matplotlib 2.2.2
- seaborn 0.9.0

## Implementations
- **Models:** logistic regression, neural network (3 FC layers), and convolutional neural network (3 Conv layers + 2 FC layers)  
- **Optimizers:** SGDNestrov, AdaGrad, RMSProp, AdaDelta, and Adam
- **Dataset:** MNIST and CIFAR10
- **With/without dropout**
- **Same weight initialization for all models and optimizers**
- **Nonlinear decreasing learning rate control**
- **Objective functions:** L2 norm for logistic regression, and softmax-cross entropy with regularization term for neural network and convolutional neural network
- **Whitening:** option only for Cifar10

## Multilayer Neural Networks Training Cost on MNIST
<p align="center">
<img src="https://user-images.githubusercontent.com/37034031/57379617-36f3b680-71e2-11e9-9ae2-156bf79dfcbf.png" width=800>
</p>

## Convolutional Neural Networks Training Cost on CIFAR10
<p align="center">
<img src="https://user-images.githubusercontent.com/37034031/57379656-4ffc6780-71e2-11e9-8b34-e40e130c2b08.png" width=800>
</p>

## Documentation
### Dataset
This implementation uses MNIST and CIFAR10 dataset. Both datasets can be downloaded automatically.  

### Directory Hierarchy
``` 
.
│   Adam-TensorFlow
│   ├── src
│   │   ├── cache.py
│   │   ├── cifar10.py
│   │   ├── dataset.py
│   │   ├── download.py
│   │   ├── main.py
│   │   ├── mnist.py
│   │   ├── models.py
│   │   ├── solver.py
│   │   ├── tensorflow_utils.py
│   │   └── utils.py
│   Data
│   ├── mnist
│   └── cifar10

```  

### Training
Use `main.py` to train the ls. Example usage:
```
python main.py
```
- `gpu_index`: gpu index if you have multiple gpus, default: `0`  
- `model`: network model in [logistic|neural_network|cnn], default: `cnn`
- `is_train`: training or test mode, default: `False (test mode)`  
- `batch_size`: batch size for one iteration, default: `128`
- `is_train`: training or inference mode, default: `True`
- `is_whiten`: whitening for CIFAR10 dataset, default: `False`
- `learning_rate`: initial learning rate for optimizer, default: `1e-3`  
- `weight_decay`: weight decay for model to handle overfitting, default: `1e-4`
- `epoch`: number of epochs, default: `200`  
- `print_freq`: print frequency for loss information, default: `50`  
- `load_model`: folder of saved model that you wish to continue training, (e.g. 20190427-1109), default: `None`  

### Test
Use `main.py` to test the models. Example usage:
```
python main.py --is_train=False --load_model=folder/you/wish/to/test/e.g./20190427-1109
```  
Please refer to the above arguments.

### Tensorboard Visualization
#### Logistic Regression on MNIST
The visualizaiton includes training (batch) and validation accuracy for each epoch and loss information during training processs.  

<p align='center'>
  <img src="https://user-images.githubusercontent.com/37034031/57382489-0282f900-71e8-11e9-8eb8-63936be60ddb.png" width=1000)
</p>  

#### Neural Network on MNIST  
The visualizaiton includes training (batch) and validation accuracy for each epoch, and total loss, data loss, and regularization term during training are also included. 

<p align='center'>
  <img src="https://user-images.githubusercontent.com/37034031/57383094-2a269100-71e9-11e9-8d0f-5fa7868d622f.png" width=1000)
</p>  
  
### Convolutional Neural Network on CIFAR10
The visualizaiton includes training (batch) and validation accuracy for each epoch, and total loss, data loss, and regularization term during training are also included. 

<p align='center'>
  <img src="https://user-images.githubusercontent.com/37034031/57383329-943f3600-71e9-11e9-8e48-547df8a021be.png" width=1000)
</p>  

### Convolutions
- Adam is the most stable optimizer for all models  
- RMSProp is comparable with the Adam, RMSProp maybe better for some models and datasets  
- Dropout gives the instability for the training process, but it is helpful to handle overfitting  
- The performance of the whitening (data preprocessing) for Cifar10 is worse than subtracting mean data-preprocessing  
- Whitening for training data and subtracting mean for test data improves performance at least 10%, but the reason is not thoroughly analyzed (I will further figure out what is the reason)
  
### Citation
```
  @misc{chengbinjin2019Adam,
    author = {Cheng-Bin Jin},
    title = {Adam-Analysis-TensorFlow},
    year = {2019},
    howpublished = {\url{https://github.com/ChengBinJin/Adam-Analysis-TensorFlow},
    note = {commit xxxxxxx}
  }
```
