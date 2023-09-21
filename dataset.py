import logging
import os
import sys
from io import open
import json
import csv
import glob
import tqdm
from typing import List
import torch
from transformers import PreTrainedTokenizer
from torch.utils.data import TensorDataset

logger = logging.getLogger(__name__)

class SingleInput(object):
    """A single training/test example for multiple choice"""

    def __init__(self, example_id, contexts, endings, label=None):
        """Constructs a InputExample.

        Args:
            example_id: Unique id for the example.
            contexts: list of str. The untokenized text of the first sequence (context of corresponding question).
            question: string. The untokenized text of the second sequence (question).
            endings: list of str. multiple choice's options. Its length must be equal to contexts' length.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.example_id = example_id
        self.contexts = contexts
        self.endings = endings
        self.label = label
    
class InputFeatures(object):
    def __init__(self,
                 example_id,
                 choices_features,
                 label

    ):
        self.example_id = example_id
        self.choices_features = [
            {
                'input_ids': input_ids,
                'input_mask': input_mask,
                'segment_ids': segment_ids
            }
            for input_ids, input_mask, segment_ids in choices_features
        ]
        self.label = label
        
        
class DataProcessor(object):
    """Base class for data converters for multiple choice data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_test_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the test set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()


class MuTualProcessor(DataProcessor):
    """Processor for the MuTual data set."""

    def get_train_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {} train".format(data_dir))
        file = os.path.join(data_dir, 'train')
        file = self._read_txt(file)
        return self._create_examples(file, 'train')

    def get_dev_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {} dev".format(data_dir))
        file = os.path.join(data_dir, 'dev')
        file = self._read_txt(file)
        return self._create_examples(file, 'dev')

    def get_test_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {} test".format(data_dir))
        file = os.path.join(data_dir, 'test')
        file = self._read_txt(file)
        return self._create_examples(file, 'test')

    def get_labels(self):
        """See base class."""
        return ["0", "1", "2", "3"]

    def _read_txt(self, input_dir):
        lines = []
        files = glob.glob(input_dir + "/*txt")
        for file in tqdm.tqdm(files, desc="read files"):
            with open(file, 'r', encoding='utf-8') as fin:
                data_raw = json.load(fin)
                data_raw["id"] = file
                lines.append(data_raw)
        return lines


    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (_, data_raw) in enumerate(lines):
            id = "%s-%s" % (set_type, data_raw["id"])
            article = data_raw["article"]

            truth = str(ord(data_raw['answers']) - ord('A'))
            options = data_raw['options']

            examples.append(
                SingleInput(
                    example_id=id,
                    contexts=[article, article, article, article], # this is not efficient but convenient
                    endings=[options[0], options[1], options[2], options[3]],
                    label=truth))
        return examples


def convert_examples_to_features(
    examples: List[SingleInput],
    label_list: List[str],
    max_length: int,
    tokenizer: PreTrainedTokenizer,
    pad_token_segment_id=0,
    pad_token=0,
    mask_padding_with_zero=True,
) -> List[InputFeatures]:
    """
    Loads a data file into a list of `InputFeatures`
    """

    label_map = {label : i for i, label in enumerate(label_list)}

    features = []
    for (ex_index, example) in tqdm.tqdm(enumerate(examples), desc="convert examples to features"):
        if ex_index % 10000 == 0:
            logger.info("Writing example %d of %d" % (ex_index, len(examples)))
        choices_features = []
        for ending_idx, (context, ending) in enumerate(zip(example.contexts, example.endings)):
            text_a = context
            # text_a = ""
            # if example.question.find("_") != -1:
            #     # this is for cloze question
            #     text_b = example.question.replace("_", ending)
            # else:
            text_b = ending

            inputs = tokenizer.encode_plus(
                text_a,
                text_b,
                add_special_tokens=True,
                max_length=max_length,
            )
            if 'num_truncated_tokens' in inputs and inputs['num_truncated_tokens'] > 0:
                logger.info('Attention!You are poping response,'
                        'you need to try to use a bigger max seq length!')

            input_ids, token_type_ids = inputs["input_ids"], inputs["token_type_ids"]

            # The mask has 1 for real tokens and 0 for padding tokens. Only real
            # tokens are attended to.
            attention_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

            # Zero-pad up to the sequence length.
            padding_length = max_length - len(input_ids)

            input_ids = input_ids + ([pad_token] * padding_length)
            attention_mask = attention_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
            token_type_ids = token_type_ids + ([pad_token_segment_id] * padding_length)

            assert len(input_ids) == max_length
            assert len(attention_mask) == max_length
            assert len(token_type_ids) == max_length
            choices_features.append((input_ids, attention_mask, token_type_ids))


        label = label_map[example.label]

        if ex_index < 2:
            logger.info("*** Example ***")
            logger.info("race_id: {}".format(example.example_id))
            for choice_idx, (input_ids, attention_mask, token_type_ids) in enumerate(choices_features):
                logger.info("choice: {}".format(choice_idx))
                logger.info("input_ids: {}".format(' '.join(map(str, input_ids))))
                logger.info("attention_mask: {}".format(' '.join(map(str, attention_mask))))
                logger.info("token_type_ids: {}".format(' '.join(map(str, token_type_ids))))
                logger.info("label: {}".format(label))

        features.append(
            InputFeatures(
                example_id=example.example_id,
                choices_features=choices_features,
                label=label,
            )
        )

    return features

def load_and_cache_examples(args, task, tokenizer, evaluate=False, test=False):
    processor = MuTualProcessor()
    # Load data features from cache or dataset file
    if evaluate:
        cached_mode = 'dev'
    elif test:
        cached_mode = 'test'
    else:
        cached_mode = 'train'
    assert (evaluate == True and test == True) == False
    
    cached_features_file = os.path.join(args.data_dir, 'cached_{}_{}_{}'.format(
        cached_mode,
        list(filter(None, args.model_name.split('/'))).pop(),
        str(args.max_seq_length)))
    
    if os.path.exists(cached_features_file) and not args.overwrite_cache:
        logger.info("Loading features from cached file %s", cached_features_file)
        features = torch.load(cached_features_file)
    else:
        logger.info("Creating features from dataset file at %s", args.data_dir)
        label_list = processor.get_labels()
        if evaluate:
            examples = processor.get_dev_examples(args.data_dir)
        elif test:
            examples = processor.get_test_examples(args.data_dir)
        else:
            examples = processor.get_train_examples(args.data_dir)
        logger.info("Training number: %s", str(len(examples)))
        features = convert_examples_to_features(
            examples,
            label_list,
            args.max_seq_length,
            tokenizer,
            pad_token_segment_id=0
        )

    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor(select_field(features, 'input_ids'), dtype=torch.long)
    all_input_mask = torch.tensor(select_field(features, 'input_mask'), dtype=torch.long)
    all_segment_ids = torch.tensor(select_field(features, 'segment_ids'), dtype=torch.long)
    all_label_ids = torch.tensor([f.label for f in features], dtype=torch.long)
    

    dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
    
    if cached_mode == 'train':
        val_split = args.val_split
        split_index = int(len(all_input_ids) * val_split)

        train_input_ids, val_input_ids = all_input_ids[split_index:], all_input_ids[:split_index]
        train_input_mask, val_input_mask = all_input_mask[split_index:], all_input_mask[:split_index]
        train_segment_ids, val_segment_ids = all_segment_ids[split_index:], all_segment_ids[:split_index]
        train_label_ids, val_label_ids = all_label_ids[split_index:], all_label_ids[:split_index]

        train_dataset = TensorDataset(train_input_ids, train_input_mask, train_segment_ids, train_label_ids)
        val_dataset = TensorDataset(val_input_ids, val_input_mask, val_segment_ids, val_label_ids)
        
        return train_dataset, val_dataset

        
    return dataset, None

def select_field(features, field):
    return [
        [
            choice[field]
            for choice in feature.choices_features
        ]
        for feature in features
    ]

processors = {
    "mutual": MuTualProcessor,
}

MULTIPLE_CHOICE_TASKS_NUM_LABELS = {
    "mutual", 4,
}