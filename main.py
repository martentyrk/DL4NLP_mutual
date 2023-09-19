import argparse

def parseArgs():
    """
    Function to parse user input argument
    """
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--model_name", default=None, type=str, required=True,
                        help="name of pre-trained-language model")
    parser.add_argument("--num_classes", default=None, type=str, required=True,
                        help="number of classes")
    parser.add_argument("--train_data_path", default=None, type=str, required=True,
                        help="path of training data")
    parser.add_argument("--val_data_path", default=None, type=str, required=True,
                        help="path of validation data")
    parser.add_argument("--test_data_path", default=None, type=str, required=True,
                        help="path of test data")
    parser.add_argument("--batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument("--num_workers", default=4, type=int,
                        help="number of cpus/gpus to run on parallel")
    parser.add_argument("--checkpoint_path", default=None, type=str, required=True,
                        help="path to store checkpoints")
    parser.add_argument("--max_epochs", default=1, type=int, required=False,
                        help="Maximum number of epochs, most likely the number of epochs")

    args = parser.parse_args()
    return args

args = parseArgs()
print(args)
model, result = fine_tune(args)

if __name__ == "__main__":
    main()