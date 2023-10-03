import argparse
# from bert import fine_tune
from bert_new import fine_tune


def parseArgs():
    """
    Function to parse user input argument
    """
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--model_name", default="bert-base-uncased", type=str, required=False,
                        help="name of pre-trained-language model")
    parser.add_argument("--data_dir", default='data_short/mutual/', type=str, required=False,
                        help="path of training data")
    parser.add_argument("--batch_size", default=64, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument("--num_workers", default=4, type=int,
                        help="number of cpus/gpus to run on parallel")
    parser.add_argument("--checkpoint_path", default='checkpoint', type=str, required=False,
                        help="path to store checkpoints")
    parser.add_argument("--max_epochs", default=10, type=int, required=False,
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

    # args for regularizers
    parser.add_argument('--contrastive_margin', type=float, default=0.5,
                        help='Margin for the contrastive loss.')
    parser.add_argument("--use_contrastive", default=True, type=bool,
                        help="Use the contrastive learning-based regularizer")
    parser.add_argument("--contrastive_weight", default=0.33, type=float,
                        help="Weight for the contrastive regularizer")
    parser.add_argument("--use_correlation", default=True, type=bool,
                        help="Use the correlation matrix-based regularizer")
    parser.add_argument("--correlation_weight", default=0.33, type=float,
                        help="Weight for the correlation regularizer")
    parser.add_argument("--crossentropy_weight", default=0.33, type=float,
                        help="Weight for the correlation regularizer")
    args = parser.parse_args()
    return args

def main():
  args = parseArgs()
  print(args)
  model, result = fine_tune(args)

if __name__ == "__main__":
    main()