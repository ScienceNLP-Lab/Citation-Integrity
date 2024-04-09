# Assessing Citation Integrity in Biomedical Publications: Annotation and NLP Models

This repository contains data and code for the paper "[Assessing Citation Integrity in Biomedical Publications: Corpus Annotation and NLP Models]."

## Data

Data is in the same format as the SciFact dataset [(schema information)](https://github.com/allenai/scifact/blob/master/doc/data.md). The dataset is split into 3 subsets: training (70 reference articles), development (10 articles), and test (20 articles).

## Code

Code was adapted from [MultiVerS](https://github.com/dwadden/multivers/tree/main).

### Model Training

Retrieve the top x sentences from the reference article using BM25+T5reranker (see bm25t5.ipynb).
Ensure that the training, validation, and test data are placed under the folder "./data_train/target/scifact_10/," including files such as claims_train.jsonl, claims_dev.jsonl, and claims_test.jsonl.

```
CUDA_LAUNCH_BLOCKING=1 TOKENIZERS_PARALLELISM=false python train_target.py --dataset scifact_10 --gpus=1 > ./output.txt 2>&1
```

### Downloading starting checkpoints

```
python get_checkpoint.py
```

### Running inference with trained models

Download our best model from [Google Drive](https://drive.google.com/drive/u/0/folders/11b6Z8iv2FXObWmLaqfYzgUQsaL4QgTT2?q=parent:11b6Z8iv2FXObWmLaqfYzgUQsaL4QgTT2).

```
python multivers/predict.py --{checkpoint_path} --{output_file} --{input_file} --{corpus_file}
```




