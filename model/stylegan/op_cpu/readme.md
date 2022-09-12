Code from [rosinality-stylegan2-pytorch-cp](https://github.com/senior-sigan/rosinality-stylegan2-pytorch-cpu)

Scripts to convert rosinality/stylegan2-pytorch to the CPU compatible format

If you would like to use CPU for testing or have a problem regarding the cpp extention (fused and upfirdn2d), please make the following changes:

Change `model.stylegan.op`  to `model.stylegan.op_cpu`
https://github.com/williamyang1991/VToonify/blob/01b383efc00007f9b069585db41a7d31a77a8806/util.py#L14

https://github.com/williamyang1991/VToonify/blob/01b383efc00007f9b069585db41a7d31a77a8806/model/simple_augment.py#L12

https://github.com/williamyang1991/VToonify/blob/01b383efc00007f9b069585db41a7d31a77a8806/model/stylegan/model.py#L11
