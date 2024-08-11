# Watermelon

## Introduction

This project is based on [crf0409/watermelon_eval](https://github.com/crf0409/watermelon_eval), reimplemented with PyTorch under [CC-BY-NC-SA 4.0 License](LICENSE). You can get the original dataset from the link above.

## File Structure

- `example-main.py`: Original main script from @crf0409 with Tensorflow.
- `clean.py`: Run it to clean the original dataset. You may have to modify the path in the script.
- `preprocess.py`: Preprocess the dataset for training and inference.
- `train.py`: Train the model.

## Usage

### Preparation

Download the dataset from [crf0409/watermelon_eval](https://github.com/crf0409/watermelon_eval), which provides links to IEEE DataPort and Baidu Netdisk. Unzip and copy to the repository root (rename the folder to `datasets` is recommended).

(Recommended) Create a virtual environment and install the dependencies:

```bash
pip install -r requirements.txt
```

### Clean the Dataset

Run `clean.py` to [clean](clean.py) the original dataset (you may have to modify the path in the script).

```bash
python clean.py
```

### Do Preprocessing

Run [`preprocess.py`](preprocess.py) to avoid duplicated preprocessing is useful to accelerate training.

```bash
python preprocess.py --data_dir /path/to/cleaned
```

### Train

Run [`train.py`](train.py) to train the model.

```bash
python train.py
```
