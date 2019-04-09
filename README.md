# A deep learning framework for activity recognition

## Requirements
- Python 3.5
- PyTorch
- [visdom](https://github.com/facebookresearch/visdom)
- [torchnet](https://github.com/pytorch/tnt)
- other required packages are listed in `requirements.txt`

## How to use
1. Start the visdom server by
python -m visdom
2. Train the network 
```commandline
python main.py --dataset_dir <main datasets path> --mode train --model_name HCN --dataset_name <dataset name> --num <experiment Nr.>
```
3. Test the network
```commandline
python main.py --dataset_dir <main datasets path> --mode test --load True --model_name HCN --dataset_name <dataset name> --num <experiment Nr.>
```
## Data pre-processing
The raw data should be transformed into numpy array (memmap format).
For example for NTU RGB+D dataset:
```commandline
python ./loader/ntu_gendata.py --data_path <main datasets path> --out_folder <sub-folder to save the transformed dataset>
```
