# Assessing Citation Integrity in Biomedical Publications: Annotation and NLP Models

This folder will contain the citation accuracy classification code for the paper "[Assessing Citation Integrity in Biomedical Publications: Corpus Annotation and NLP Models]." Partial code was borrowed from the folder (https://github.com/dwadden/multivers/tree/main).

## Run the script to train the model on the dataset

Retrieve the top x sentences from the reference article using BM25+T5reranker (see bm25t5.ipynb).
Ensure that the training, validation, and test data are placed under the folder "./data_train/target/scifact_10/," including files such as claims_train.jsonl, claims_dev.jsonl, and claims_test.jsonl.

```
CUDA_LAUNCH_BLOCKING=1 TOKENIZERS_PARALLELISM=false python train_target.py --dataset scifact_10 --gpus=1 > ./output.txt 2>&1
```

## Download starting checkpoints

```
python get_checkpoint.py
```

### Run inference with trained models

```
python multivers/predict.py --{checkpoint_path} --{output_file} --{input_file} --{corpus_file}
```

## Download our best model

https://drive.google.com/drive/u/0/folders/11b6Z8iv2FXObWmLaqfYzgUQsaL4QgTT2?q=parent:11b6Z8iv2FXObWmLaqfYzgUQsaL4QgTT2