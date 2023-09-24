import argparse
from bert import fine_tune



def parseArgs():
    """
    Function to parse user input argument
    """
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--model_name", default="bert-base-uncased", type=str, required=False,
                        help="name of pre-trained-language model, use either: TODBERT/TOD-BERT-MLM-V1, bert-base-uncased")
    parser.add_argument("--data_dir", default='data/mutual/', type=str, required=False,
                        help="path of training data")
    parser.add_argument("--batch_size", default=1, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument("--num_workers", default=4, type=int,
                        help="number of cpus/gpus to run on parallel")
    parser.add_argument("--checkpoint_path", default='checkpoint', type=str, required=False,
                        help="path to store checkpoints")
    parser.add_argument("--max_epochs", default=1, type=int, required=False,
                        help="Maximum number of epochs, most likely the number of epochs")
    parser.add_argument("--seed", default=42, type=int, required=False,
                        help="Seed value")
    parser.add_argument("--max_seq_length", default=128, type=int,
                        help="The maximum total input sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")
    parser.add_argument("--task_name", default='mutual', type=str, required=False,
                        help="The name of the task: mutual ")
    parser.add_argument("--val_split", default=0.1, type=float, required=False,
                        help="Size of the validation split from training")
    parser.add_argument("--device", default='cpu', type=str, required=False,
                        help="Size of the validation split from training")
    parser.add_argument("--freeze_lm", default=True, type=bool, required=False,
                        help="Freeze all layers except last")
    args = parser.parse_args()
    return args

def main():
  args = parseArgs()
  print(args)
  model, result = fine_tune(args)

if __name__ == "__main__":
    main()