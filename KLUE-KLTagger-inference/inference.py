""" Usage
$ python inference.py
"""

import argparse
import os
import tarfile

import sklearn.metrics
import torch
from dataloader import KlueDpDataLoader
from model import AutoModelforDp
from transformers import AutoConfig, AutoTokenizer
from utils import flatten_prediction_and_labels, get_dp_labels
import numpy as np




def load_model(model_dir, args):


    config = AutoConfig.from_pretrained(os.path.join(model_dir, "config.json"))
    model = AutoModelforDp(config, args)
    model.load_state_dict(torch.load(os.path.join(model_dir, "dp-model.bin"), map_location='cpu'))
    return model


@torch.no_grad()
def inference(data_dir, model_dir, output_dir, args):
    # device setup
    num_gpus = torch.cuda.device_count()
    use_cuda = num_gpus > 0
    device = torch.device("cuda" if use_cuda else "cpu")

    # load model
    model = load_model(model_dir, args)
    model.to(device)
    model.eval()

    # load-DP-test
    kwargs = {"num_workers": num_gpus, "pin_memory": True} if use_cuda else {}
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    klue_dp_dataset = KlueDpDataLoader(args, tokenizer, data_dir)
    klue_dp_test_loader = klue_dp_dataset.get_test_dataloader(
        args.test_filename, **kwargs
    )

    # inference
    predictions = []
    labels = []
    for i, batch in enumerate(klue_dp_test_loader):
        input_ids, masks, ids, max_word_length = batch
        input_ids = input_ids.to(device)
        attention_mask, bpe_head_mask, bpe_tail_mask, mask_e, mask_d = (
            mask.to(device) for mask in masks
        )
        head_ids, type_ids, pos_ids, pos_ids2, pos_ids3 = (id.to(device) for id in ids)
        batch_size, _ = head_ids.size()
        batch_index = torch.arange(0, batch_size).long()

        out_arc, out_type = model(
            bpe_head_mask,
            bpe_tail_mask,
            pos_ids,
            pos_ids2,
            pos_ids3,
            head_ids,
            max_word_length,
            mask_e,
            mask_d,
            batch_index,
            input_ids,
            attention_mask,
        )




        heads = torch.argmax(out_arc, dim=2)

        types = torch.argmax(out_type, dim=2)

        prediction = (heads, types)
        predictions.append(prediction)

        # predictions are valid where labels exist
        label = (head_ids, type_ids)
        labels.append(label)

    head_preds, type_preds, head_labels, type_labels = flatten_prediction_and_labels(predictions, labels)



    labels_list = get_dp_labels()
    lines_list = []
    guid=""
    with open("C:\\Users\\kihoon\\Desktop\\KLUE-KLTagger-inference\\data\\dp-v1.1_test.tsv", "r", encoding="utf8") as f2:
        sent_id=-1
        for line in f2:
            if line == "" or line == "\n" or line == "\t":
                continue

            if line.startswith("#"):
                parsed = line.strip().split("\t")
                if len(parsed) != 2:  # metadata line about dataset
                    continue
                else:
                    sent_id += 1
                    text = parsed[1].strip()
                    guid = parsed[0].replace("##", "").strip()
            else:
                token_list=line.split("\t")
                token_list[4] = head_preds[int(token_list[0]) - 1]
                token_list[5] = labels_list[int(type_preds[int(token_list[0]) - 1])]

                line = "%d\t%s\t%s\t%s\t%d\t%s\n" % (
                    int(token_list[0]), token_list[1], token_list[2], token_list[3], token_list[4], token_list[5])

                lines_list.append(line)

    with open("C:\\Users\\kihoon\\Desktop\\KLUE-KLTagger-inference\\data\\dp-v1.1_test.tsv", "w", encoding="utf8") as f3:
        lines = []
        colum_name = "## 칼럼명 : INDEX	WORD_FORM	LEMMA	POS	HEAD	DEPREL"
        f3.write(colum_name)
        f3.write("\n")
        line2 = "##sentence:\t" + guid + "\n"
        f3.write(line2)
        for line in lines_list:
            f3.write(line)




if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Container environment
    parser.add_argument(
        "--data_dir", type=str,
        #default=os.environ.get("SM_CHANNEL_EVAL", "/data")
        default="C:\\Users\\kihoon\\Desktop\\KLUE-KLTagger-inference\\data"
    )
    parser.add_argument(
        "--model_dir", type=str,
        default="C:\\Users\\kihoon\\Desktop\\KLUE-KLTagger-inference\\model"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=os.environ.get("SM_OUTPUT_DATA_DIR", "/output"),
    )

    # inference arguments
    """
    parser.add_argument(
        "--model_tar_file",
        type=str,
        default="klue-dp.tar.gz",
        help="it needs to include all things for loading baseline model & tokenizer, \
             only supporting transformers.AutoModelForSequenceClassification as a model \
             transformers.XLMRobertaTokenizer or transformers.BertTokenizer as a tokenizer",
    )
    """
    parser.add_argument(
        "--test_filename",
        default="dp-v1.1_test.tsv",
        type=str,
        help="Name of the test file (default: klue-dp-v1.1_test.tsv)",
    )
    parser.add_argument("--eval_batch_size", default=64, type=int)

    # model-specific arguments
    parser = AutoModelforDp.add_arguments(parser)

    # parse args
    args = parser.parse_args()

    data_dir = args.data_dir
    model_dir = args.model_dir
    output_dir = args.output_dir

    inference(data_dir, model_dir, output_dir, args)

