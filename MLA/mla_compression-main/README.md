# TP

This TP is composed of three mini-projects. Each of these projects correspond to a different compression task.

## task 1 - batch-normalization folding

The goal of this project is to fold batch-normalization layers. Such layers are only useful during training. At inference, they slow down the process without offering more expressivity. In order to fold a BN layer we need to se where they occur in the model. For easier model visualization you should install [netron](https://netron.app/).

## task 2 - pruning

Pruning consists in removing neurons (or channels) in the graph. To do so, we define a pruning criterion which meausres the importance of neurons within a layer. Then we set the coresponding kernels to zero. Finally, we edit the graph to actually remove these neurons and fine-tuning the resulting model.

## task 3 - quantization

Quantization consists in converting floating point (rational) operations into fixed point (integer) operations. To do so, we need to round the values in the conv layers.