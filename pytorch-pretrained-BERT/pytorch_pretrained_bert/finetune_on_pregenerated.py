import os
import torch
import time
from torch.utils.data import DataLoader, Dataset, RandomSampler
from torch.utils.data.distributed import DistributedSampler
from pytorch_pretrained_bert.modeling import BertForPreTraining
from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.optimization import BertAdam
import torch.multiprocessing as multiprocessing

from argparse import ArgumentParser
from pathlib import Path

import logging
import json
import random
import numpy as np
from collections import namedtuple
from tempfile import TemporaryDirectory
import pickle

from tqdm import tqdm



InputFeatures = namedtuple("InputFeatures", "input_ids input_mask segment_ids lm_label_ids is_next masked_lm_positions")

log_format = '%(asctime)-10s: %(message)s'
logging.basicConfig(level=logging.INFO, format=log_format)




def convert_example_to_features(example, tokenizer, max_seq_length):
    tokens = example["tokens"]
    segment_ids = example["segment_ids"]
    is_random_next = example["is_random_next"]
    masked_lm_positions = example["masked_lm_positions"]
    masked_lm_labels = example["masked_lm_labels"]

    assert len(tokens) == len(segment_ids) <= max_seq_length  # The preprocessed data should be already truncated
    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    masked_label_ids = tokenizer.convert_tokens_to_ids(masked_lm_labels)

    input_array = np.zeros(max_seq_length, dtype=np.int)
    input_array[:len(input_ids)] = input_ids

    mask_array = np.zeros(max_seq_length, dtype=np.bool)
    mask_array[:len(input_ids)] = 1

    segment_array = np.zeros(max_seq_length, dtype=np.bool)
    segment_array[:len(segment_ids)] = segment_ids

    lm_label_array = np.full(max_seq_length, dtype=np.int, fill_value=-1)
    lm_label_array[masked_lm_positions] = masked_label_ids

    features = InputFeatures(input_ids=input_array,
                             input_mask=mask_array,
                             segment_ids=segment_array,
                             lm_label_ids=lm_label_array,
                             is_next=is_random_next,
                             masked_lm_positions=masked_lm_positions)
    return features



class PregeneratedDataset(Dataset):
    def __init__(self, training_path, epoch, tokenizer, num_data_epochs, reduce_memory=False):
        self.vocab = tokenizer.vocab
        self.tokenizer = tokenizer
        self.epoch = epoch
        self.data_epoch = epoch % num_data_epochs
        data_file = training_path / f"epoch_{self.data_epoch}.json"
        metrics_file = training_path / f"epoch_{self.data_epoch}_metrics.json"
        assert data_file.is_file() and metrics_file.is_file()
        metrics = json.loads(metrics_file.read_text())
        num_samples = metrics['num_training_examples']
        seq_len = metrics['max_seq_len']
        self.temp_dir = None
        self.working_dir = None
        if reduce_memory:
            self.temp_dir = TemporaryDirectory()
            self.working_dir = Path(self.temp_dir.name)
            input_ids = np.memmap(filename=self.working_dir / 'input_ids.memmap',
                                  mode='w+', dtype=np.int32, shape=(num_samples, seq_len))
            input_masks = np.memmap(filename=self.working_dir / 'input_masks.memmap',
                                    shape=(num_samples, seq_len), mode='w+', dtype=np.bool)
            segment_ids = np.memmap(filename=self.working_dir / 'input_masks.memmap',
                                    shape=(num_samples, seq_len), mode='w+', dtype=np.bool)
            lm_label_ids = np.memmap(filename=self.working_dir / 'lm_label_ids.memmap',
                                     shape=(num_samples, seq_len), mode='w+', dtype=np.int32)
            mask_positions = np.full(shape=(num_samples, seq_len), dtype=np.int32, fill_value=-1)
            lm_label_ids[:] = -1
            is_nexts = np.memmap(filename=self.working_dir / 'is_nexts.memmap',
                                 shape=(num_samples,), mode='w+', dtype=np.bool)
        else:
            input_ids = np.zeros(shape=(num_samples, seq_len), dtype=np.int32)
            input_masks = np.zeros(shape=(num_samples, seq_len), dtype=np.bool)
            segment_ids = np.zeros(shape=(num_samples, seq_len), dtype=np.bool)
            lm_label_ids = np.full(shape=(num_samples, seq_len), dtype=np.int32, fill_value=-1)
            is_nexts = np.zeros(shape=(num_samples,), dtype=np.bool)
            mask_positions = np.full(shape=(num_samples, seq_len), dtype=np.int32, fill_value=-1)
        logging.info(f"Loading training examples for epoch {epoch}")

        with data_file.open() as f:
            for i, line in enumerate(tqdm(f, total=num_samples, desc="Training examples")):
                line = line.strip()
                example = json.loads(line)
                features = convert_example_to_features(example, tokenizer, seq_len)
                input_ids[i] = features.input_ids
                segment_ids[i] = features.segment_ids
                input_masks[i] = features.input_mask
                lm_label_ids[i] = features.lm_label_ids
                is_nexts[i] = features.is_next
                for j in range(len(features.masked_lm_positions)):
                    mask_positions[i][j] = features.masked_lm_positions[j]
            # assert i == num_samples - 1  # Assert that the sample count metric was true
        logging.info("Loading complete!")
        self.num_samples = num_samples
        self.seq_len = seq_len
        self.input_ids = input_ids
        self.input_masks = input_masks
        self.segment_ids = segment_ids
        self.lm_label_ids = lm_label_ids
        self.is_nexts = is_nexts
        self.mask_positions = mask_positions

    def __len__(self):
        return self.num_samples

    def __getitem__(self, item):
        return (torch.tensor(self.input_ids[item].astype(np.int64)),
                torch.tensor(self.input_masks[item].astype(np.int64)),
                torch.tensor(self.segment_ids[item].astype(np.int64)),
                torch.tensor(self.lm_label_ids[item].astype(np.int64)),
                torch.tensor(self.is_nexts[item].astype(np.int64)),
                torch.tensor(self.mask_positions[item].astype(np.int64)))



def main():

    parser = ArgumentParser()
    parser.add_argument('--pregenerated_data', type=Path, required=True)

    parser.add_argument('--dist_url', type=str, default="tcp://172.31.38.122:23456")
    parser.add_argument('--rank', type=int, default=0)
    parser.add_argument('--output_dir', type=Path, required=True)
    parser.add_argument('--use_all_gpus', action="store_true")
    parser.add_argument('--world_size', type=int, default=1)

    parser.add_argument('--output_file', type=str, default = "pytorch_model.bin")

    parser.add_argument("--bert_model", type=str, required=True, help="Bert pre-trained model selected in the list: bert-base-uncased, "
                             "bert-large-uncased, bert-base-cased, bert-base-multilingual, bert-base-chinese.")

    parser.add_argument("--no_sentence_loss", action="store_true",
                        help="Whether not to use sentence level loss.")

    parser.add_argument('--tokeniser', type=str, default="vocab.txt")

    parser.add_argument("--do_lower_case", action="store_true")

    parser.add_argument("--reduce_memory", action="store_true",
                        help="Store training data as on-disc memmaps to massively reduce memory usage")

    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs to train for")

    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")

    parser.add_argument("--no_cuda",
                        action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")


    parser.add_argument('--training',
                        type = bool,
                        default = True,
                        help = "Whether to train the model or not")

    parser.add_argument("--train_batch_size",
                        default=32,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument('--fp16',
                        action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")

    parser.add_argument('--loss_scale',
                        type=float, default=0,
                        help="Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
                        "0 (default value): dynamic loss scaling.\n"
                        "Positive power of 2: static loss scaling value.\n")

    parser.add_argument("--warmup_proportion",
                        default=0.1,
                        type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10%% of training.")

    parser.add_argument("--learning_rate",
                        default=3e-5,
                        type=float,
                        help="The initial learning rate for Adam.")

    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")

    parser.add_argument('--verbose',
                        action='store_true',
                        help="Whether to print more details along the way")

    parser.add_argument('--tensorboard',
                        action='store_true',
                        help="Whether to use Tensorboard ")

    parser.add_argument('--save',
                        type=bool,
                        default=True,
                        help="Whether to save")
    parser.add_argument('--bert_finetuned', type = str, default = None, help='Model finetuned to use instead of pretrained models')



    args = parser.parse_args()

    # if args.tensorboard :
    #     from modeling import BertForPreTraining








    assert args.pregenerated_data.is_dir(), \
        "--pregenerated_data should point to the folder of files made by pregenerate_training_data.py!"

    samples_per_epoch = []
    for i in range(args.epochs):
        epoch_file = args.pregenerated_data / f"epoch_{i}.json"
        metrics_file = args.pregenerated_data / f"epoch_{i}_metrics.json"
        if epoch_file.is_file() and metrics_file.is_file():
            metrics = json.loads(metrics_file.read_text())
            samples_per_epoch.append(metrics['num_training_examples'])
        else:
            if i == 0:
                exit("No training data was found!")
            print(f"Warning! There are fewer epochs of pregenerated data ({i}) than training epochs ({args.epochs}).")
            print("This script will loop over the available data, but training diversity may be negatively impacted.")
            num_data_epochs = i
            break
    else:
        num_data_epochs = args.epochs

    if args.local_rank == -1 or args.no_cuda:
        n_gpu = torch.cuda.device_count()
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")


    else:
        if args.use_all_gpus:
            device = torch.device("cuda")
            n_gpu = torch.cuda.device_count()

            dp_device_ids = list(range(min(n_gpu,args.train_batch_size)))

        else:
            torch.cuda.set_device(args.local_rank)
            device = torch.device("cuda", args.local_rank)
            dp_device_ids = [args.local_rank]
            n_gpu = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        print("Initialize Process Group...")
        # torch.distributed.init_process_group(backend='nccl')
        # Number of distributed processes
        world_size = args.world_size

        # Distributed backend type 

        dist_backend = 'gloo'
        start= time.time()
        torch.distributed.init_process_group(backend=dist_backend, init_method=args.dist_url, rank=args.rank,
                                world_size=world_size)
        end = time.time()
    print('done within :', end-start)
    logging.info("device: {} n_gpu: {}, distributed training: {}, 16-bits training: {}".format(
        device, n_gpu, bool(args.local_rank != -1), args.fp16))

    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
                            args.gradient_accumulation_steps))

    ##### COMBIEN DE GPUs UTILLISER


    args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    if args.output_dir.is_dir() and list(args.output_dir.iterdir()):
        logging.warning(f"Output directory ({args.output_dir}) already exists and is not empty!")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=args.do_lower_case)
    tokenizer = BertTokenizer.from_pretrained('vocab.txt', do_lower_case=args.do_lower_case)
    total_train_examples = 0
    for i in range(args.epochs):
        # The modulo takes into account the fact that we may loop over limited epochs of data
        total_train_examples += samples_per_epoch[i % len(samples_per_epoch)]

    num_train_optimization_steps = int(
        total_train_examples / args.train_batch_size / args.gradient_accumulation_steps)
    if args.local_rank != -1:
        num_train_optimization_steps = num_train_optimization_steps // torch.distributed.get_world_size()
        # num_train_optimization_steps = num_train_optimization_ste ps // n_gpu
    # Prepare model

    try:
        model = BertForPreTraining.from_pretrained(args.bert_model, verbose = args.verbose, tokeniser = args.tokeniser,train_batch_size = args.train_batch_size, device = device)
    except:
        model = BertForPreTraining.from_pretrained(args.bert_model)

    if args.bert_finetuned is not None:
        model_dict = torch.load(args.bert_finetuned)
        model.load_state_dict(model_dict)

    if args.fp16:
        model.half()

    if args.local_rank != -1:
        # try:
        #     from apex.parallel import DistributedDataParallel as DDP
        # except ImportError:
        #     raise ImportError(
        #         "Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

        print("Initialize Model...")
        # model = DDP(model, device_ids = dp_device_ids,output_device=args.local_rank)
        model.to(device)
        model = torch.nn.parallel.DistributedDataParallel(model , device_ids = dp_device_ids, output_device = args.local_rank)
        n_gpu_used = model.device_ids
        print('number gpu used', n_gpu_used)

    elif n_gpu > 1:



        # torch.cuda.set_device(list(range(min(args.train_batch_size, n_gpu))))

        model = torch.nn.DataParallel(model)#, device_ids = list(range(min(args.train_batch_size, n_gpu))))
        n_gpu_used = model.device_ids
        print('number gpu used', n_gpu_used)
    elif n_gpu ==1:
        print("Only 1 GPU used")
        n_gpu_used = [1]
    model.to(device)

    # Prepare optimizer
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    if args.fp16:
        try:
            from apex.optimizers import FP16_Optimizer
            from apex.optimizers import FusedAdam
        except ImportError:
            raise ImportError(
                "Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

        optimizer = FusedAdam(optimizer_grouped_parameters,
                              lr=args.learning_rate,
                              bias_correction=False,
                              max_grad_norm=1.0)
        if args.loss_scale == 0:
            optimizer = FP16_Optimizer(optimizer, dynamic_loss_scale=True)
        else:
            optimizer = FP16_Optimizer(optimizer, static_loss_scale=args.loss_scale)

    else:
        optimizer = BertAdam(optimizer_grouped_parameters,
                             lr=args.learning_rate,
                             warmup=args.warmup_proportion,
                             t_total=num_train_optimization_steps)

    global_step = 0
    logging.info("***** Running training *****")
    logging.info(f"  Num examples = {total_train_examples}")
    logging.info("  Batch size = %d", args.train_batch_size)
    logging.info("  Num steps = %d", num_train_optimization_steps)
    model.train()
    for epoch in range(args.epochs):
        epoch_dataset = PregeneratedDataset(epoch=epoch, training_path=args.pregenerated_data, tokenizer=tokenizer,
                                            num_data_epochs=num_data_epochs)
        if args.local_rank == -1:
            train_sampler = RandomSampler(epoch_dataset)
        else:
            train_sampler = DistributedSampler(epoch_dataset)
        train_dataloader = DataLoader(epoch_dataset, sampler=train_sampler,num_workers=0, batch_size=args.train_batch_size, pin_memory=False)
        tr_loss = 0
        nb_tr_examples, nb_tr_steps = 0, 0

        with tqdm(total=len(train_dataloader), desc=f"Epoch {epoch}") as pbar:
            for step, batch in enumerate(train_dataloader):
                if args.training:
                    model.train()
                    batch = tuple(t.to(device, non_blocking = True) for t in batch)
                    input_ids, input_mask, segment_ids, lm_label_ids, is_next, mask_index = batch
                    if args.no_sentence_loss:
                        is_next = None
                    loss = model(input_ids, segment_ids, input_mask, lm_label_ids, is_next, mask_index)
                    if args.verbose:
                        # print('input_ids : ', input_ids)
                        #
                        # print('input_mask : ', input_mask)
                        # print('segment_ids : ', segment_ids)
                        # print('lm_label_ids : ', lm_label_ids)
                        # print('is_next : ', is_next)
                        print('loss : ',loss)

                    # if n_gpu > 1:
                    if len(n_gpu_used) >1:
                        loss = loss.mean() # mean() to average on multi-gpu.
                    if args.gradient_accumulation_steps > 1:
                        loss = loss / args.gradient_accumulation_steps

                    if args.fp16:
                        optimizer.backward(loss)
                    else:
                        # print('backwards')
                        loss.backward()
                    # print('backwards done')
                    tr_loss += loss.item()
                    nb_tr_examples += input_ids.size(0)
                    nb_tr_steps += 1
                    pbar.update(1)
                    mean_loss = tr_loss * args.gradient_accumulation_steps / nb_tr_steps
                    pbar.set_postfix_str(f"Loss: {mean_loss:.5f}")
                    if (step + 1) % args.gradient_accumulation_steps == 0:
                        if args.fp16:
                            # modify learning rate with special warm up BERT uses
                            # if args.fp16 is False, BertAdam is used that handles this automatically
                            lr_this_step = args.learning_rate * warmup_linear(global_step/num_train_optimization_steps,
                                                                              args.warmup_proportion)
                            for param_group in optimizer.param_groups:
                                param_group['lr'] = lr_this_step
                        optimizer.step()
                        optimizer.zero_grad()
                        global_step += 1
                else:
                    with torch.no_grad():
                        model.eval()
                        batch = tuple(t.to(device) for t in batch)
                        input_ids, input_mask, segment_ids, lm_label_ids, is_next = batch
                        loss = model(input_ids, segment_ids, input_mask, lm_label_ids, is_next)
                        if args.verbose:
                            print('input_ids : ', input_ids)
                            print('input_mask : ', input_mask)
                            print('segment_ids : ', segment_ids)
                            print('lm_label_ids : ', lm_label_ids)
                            print('is_next : ', is_next)
                            print('loss : ', loss)
                        if n_gpu > 1:
                            loss = loss.mean()  # mean() to average on multi-gpu.
                        if args.gradient_accumulation_steps > 1:
                            loss = loss / args.gradient_accumulation_steps

                        tr_loss += loss.item()
                        nb_tr_examples += input_ids.size(0)
                        nb_tr_steps += 1
                        pbar.update(1)
                        mean_loss = tr_loss * args.gradient_accumulation_steps / nb_tr_steps
                        pbar.set_postfix_str(f"Loss: {mean_loss:.5f}")
                        if (step + 1) % args.gradient_accumulation_steps == 0:

                                # modify learning rate with special warm up BERT uses
                                # if args.fp16 is False, BertAdam is used that handles this automatically

                            global_step += 1


    # Save a trained model
    if args.save :
        # pickle.dump(model.df,open('results.p','wb'))
        logging.info("** ** * Saving fine-tuned model ** ** * ")
        model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
        output_model_file = args.output_dir / args.output_file
        torch.save(model_to_save.state_dict(), str(output_model_file))


if __name__ == '__main__':
    # delete torch.multiprocessing.set_start_method('forkserver')
    multiprocessing = multiprocessing.get_context('spawn')
    torch.multiprocessing.set_start_method('spawn', force=True)
    print('debut')
    main()
