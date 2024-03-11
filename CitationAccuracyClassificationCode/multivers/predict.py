import sys
from tqdm import tqdm
import argparse
from pathlib import Path
from model import MultiVerSModel
import model
from data import get_dataloader
import util
import json
import data_train as dm

model.IF_TRAIN = False

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_path", required=True, type=str) 
    parser.add_argument("--input_file", type=str)
    parser.add_argument("--corpus_file", type=str)
    parser.add_argument("--output_file", required=True, type=str)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--device", default=0, type=int)
    parser.add_argument("--num_workers", default=4, type=int)
    parser.add_argument(
        "--no_nei", action="store_true", help="If given, never predict NEI."
    )
    parser.add_argument(
        "--force_rationale",
        action="store_true",
        help="If given, always predict a rationale for non-NEI.",
    )
    parser.add_argument("--debug", action="store_true")

    return parser.parse_args()


def get_predictions(args):
    model = MultiVerSModel.load_from_checkpoint(args.checkpoint_path)
    model.to(f"cuda:{args.device}")
    model.eval()
    model.freeze()
    
    # If not predicting NEI, set the model label threshold to 0.
    if args.no_nei:
        model.label_threshold = 0.0
    # Grab model hparams and override using new args, when relevant.
    hparams = model.hparams["hparams"]
    del hparams.precision  # Don' use 16-bit precision during evaluation.
    for k, v in vars(args).items():
        if hasattr(hparams, k):
            setattr(hparams, k, v)

    with open("../train_configs.json", "r") as json_file:
        deserialized_dict = json.load(json_file)
    args_as_namespace = argparse.Namespace(**deserialized_dict)
    data_module = dm.ConcatDataModule(args_as_namespace)
    data_module.setup()

    dataloader = data_module.test_dataloader()
    
    predictions_all = []
    # Make predictions.
    for batch in tqdm(dataloader):
        preds_batch = model.predict(batch, args.force_rationale)
        predictions_all.extend(preds_batch)

    return predictions_all

def format_predictions(args, predictions_all):
    # Need to get the claim ID's from the original file, since the data loader
    # won't have a record of claims for which no documents were retireved.
    claims = util.load_jsonl(args.input_file)
    
    claim_ids = [x["id"] for x in claims]
    assert len(claim_ids) == len(set(claim_ids))

    formatted = {claim: {} for claim in claim_ids}
    
    # Dict keyed by claim.
    for prediction in predictions_all:
        # Add prediction.
        formatted_entry = {
            prediction["abstract_id"]: {
                "label": prediction["predicted_label"],
                "label_probs": prediction["label_probs"],
            }
        }
        formatted[prediction["claim_id"]].update(formatted_entry)

    # Convert to jsonl.
    res = []
    for k, v in formatted.items():
        to_append = {"id": k, "evidence": v}
        res.append(to_append)

    return res


def main():
    args = get_args()

    outname = Path(args.output_file)
    predictions = get_predictions(args)

    # Save final predictions as json.
    formatted = format_predictions(args, predictions)
    util.write_jsonl(formatted, outname)


if __name__ == "__main__":
    main()
