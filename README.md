# Effecient Bearing Fault Diagnosis

## Dependencies and Installation

### Environments

- Python 3.8.10 + CUDA 11.3 (Docker NNI)
- NNI version 2.10 build from source

Let me know if your environments can run this repo

### Install requirements

``` bash
pip install -r requirements.txt
```

### If you want to use Docker

``` terminal
docker pull msranni/nni
docker images # to see docker image id
docker run -it --gpus all --name nni [DOCKER IMAGE]
```

### Install NNI from source

- Uninstall old version

``` terminal
pip uninstall nni
```

- Install from source

``` terminal
git clone https://github.com/microsoft/nni.git
cd nni
python setup.py develop
```

## Data

Data in this format

``` folder
|-- Bearing-Dataset-16x16-noise
    |-- train
    |   |-- B
    |   |   |-- B001.mat
    |   |   |-- B001.npy   
    |   |   |-- B001.jpg
    |   |   `-- ...
    |   |-- I
    |   |-- L
    |   |-- N
    |   `-- O
    `-- test
        |   |-- B
        |   |-- B418.mat
        |   |-- B418.npy   
        |   |-- B418.jpg
        |   `-- ...
        |-- I
        |-- L
        |-- N
        `-- O
```

with `.mat` is temperature generate with MATLAB, `.npy` is temperature convert to numpy format, and `.jpg` is image of bearing signal 

## NAS

- This project use [Random](https://nni.readthedocs.io/en/stable/reference/nas/strategy.html#nni.retiarii.strategy.Random) Search Stategy, other stategies can be found [here](https://nni.readthedocs.io/en/stable/nas/exploration_strategy.html)

Run NAS

``` terminal
python main.py
```

## Quantization

- This project use [QAT Quantizer](https://nni.readthedocs.io/en/stable/reference/compression/quantizer.html#qat-quantizer) for quantization, other quantizer can be found [here](https://nni.readthedocs.io/en/stable/compression/quantizer.html)

Start Quantizing

``` terminal
python quantization.py
```

If you want to quantize multiple times, modify number of times in `quanti.sh` and run

``` terminal
bash quantization.sh
```

