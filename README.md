# SkyCloud Generation and Segmentation

## Segmentation 
- [DeepLabV3](./segment/deeplabv3/)
    - [SlurmJob output](./slurm-1571171.out)

## Generation 
- [DCGAN](./generate/dcganv2/)
    - [SlurmJob results](./generate/dcganv2/results/WSISEG64/)
    - [SlurmJob output](./slurm-1570828.out)
- [Pix2Pix](./generate/pytorch-CycleGAN-and-pix2pix/)
    - [SlurmJob results](./generate/pytorch-CycleGAN-and-pix2pix/checkpoints/WSISEG_pix2pix/web/index.html)
    - [SlurmJob output](./slurm-1571169.out)

## Installation
- Pytorch Pix2Pix
    ```
    conda create -n <env> python=3.8  
    conda install pytorch==1.8.1 torchvision==0.9.1 cudatoolkit=11.1 -c pytorch -c conda-forge  
    conda install scipy
    conda install dominate Pillow numpy visdom wandb -c conda-forge
    ```
- DeepLabV3
    ```
    conda install scikit-learn
    conda install hydra-core coloredlogs -c conda-forge
    pip install hydra-optuna-sweeper
    ```
- DCGAN 
    ```
    conda install imageio matplotlib -c conda-forge
    conda install tensorflow cudatoolkit=11.1 -c conda-forge
    ```

## Usage 
Update config
```
cd ./segment/deeplabv3/
python train.py
```
```
cd ./generate/dcganv2
python dcgan.py
```
```
cd ./generate/pytorch-CycleGAN-and-pix2pix
python train.py --dataroot ../../data/WSISEG-Database-split/combined --name WSISEG_pix2pix --model pix2pix --preprocess crop --crop_size 256 --direction AtoB
```

### Evaluaiton
```
python test.py --dataroot ../../data/WSISEG-Database-split/combined --name WSISEG_pix2pix --model pix2pix --direction AtoB
```


### Prediction
```
python test.py --dataroot ../../data/WSISEG-Database-split/whole-sky-images --name WSISEG_pix2pix --model test --direction AtoB --netG unet_256 --dataset_mode single --norm batch 
```
python test.py --dataroot ../../data/WSISEG-Database/whole-sky-images --name WSISEG_pix2pix --model test --direction AtoB --netG unet_256 --dataset_mode single --norm batch 
python test.py --dataroot ../../data/WSISEG-Database/whole-sky-images/ --name WSISEG_pix2pix --model test --netG unet_256 --direction AtoB --dataset_mode single --norm batch --preprocess crop --crop_size 256
