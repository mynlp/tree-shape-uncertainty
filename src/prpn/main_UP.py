import argparse
import datetime
import math
import os
import sys
import time

import data_ptb as data
import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from model_PRPN import PRPN

# from test_phrase_grammar import test
from torch.autograd import Variable

# To quit training when loss converges.
EPS = 1e-4

# Early stop the training if dev loss does not improve for several epochs.
EARLY_STOP_EPOCHS = 5


def mylog(*args):
    print(datetime.datetime.now(), ": ", *args, file=sys.stderr)


parser = argparse.ArgumentParser(
    description="PyTorch PennTreeBank RNN/LSTM Language Model"
)
# parser.add_argument(
#    "--data", type=str, default="./data/ptb", help="location of the data corpus"
# )
# Added
parser.add_argument("--train_file", type=str, help="location of the train corpus")
parser.add_argument("--dev_file", type=str, help="location of the dev corpus")
parser.add_argument("--test_file", type=str, help="location of the test corpus")

parser.add_argument("--emsize", type=int, default=200, help="size of word embeddings")
parser.add_argument(
    "--nhid", type=int, default=400, help="number of hidden units per layer"
)
parser.add_argument("--nlayers", type=int, default=2, help="number of layers")
parser.add_argument("--lr", type=float, default=0.001, help="initial learning rate")
parser.add_argument("--weight_decay", type=float, default=1e-6, help="weight decay")
parser.add_argument("--clip", type=float, default=1.0, help="gradient clipping")
parser.add_argument("--epochs", type=int, default=100, help="upper epoch limit")
parser.add_argument(
    "--batch_size", type=int, default=64, metavar="N", help="batch size"
)
# Not used.
# parser.add_argument("--bptt", type=int, default=35, help="sequence length")
parser.add_argument(
    "--dropout",
    type=float,
    default=0.2,
    help="dropout applied to output layers (0 = no dropout)",
)
parser.add_argument(
    "--idropout",
    type=float,
    default=0.2,
    help="dropout applied to layers (0 = no dropout)",
)
parser.add_argument(
    "--rdropout",
    type=float,
    default=0,
    help="dropout applied to recurrent states (0 = no dropout)",
)
parser.add_argument(
    "--tied", action="store_true", help="tie the word embedding and softmax weights"
)
parser.add_argument("--hard", action="store_true", help="use hard sigmoid")
parser.add_argument(
    "--res", type=int, default=0, help="number of resnet block in predict network"
)
parser.add_argument("--seed", type=int, default=1111, help="random seed")
parser.add_argument("--cuda", action="store_true", help="use CUDA")
parser.add_argument(
    "--log-interval", type=int, default=100, metavar="N", help="report interval"
)
parser.add_argument(
    "--save",
    type=str,
    default="./model/model_UP.pt",
    help="path to save the final model",
)
parser.add_argument(
    "--load", type=str, default=None, help="path to save the final model"
)
parser.add_argument("--nslots", type=int, default=15, help="number of memory slots")
parser.add_argument(
    "--nlookback",
    type=int,
    default=5,
    help="number of look back steps when predict gate",
)
parser.add_argument(
    "--resolution", type=float, default=0.1, help="syntactic distance resolution"
)
parser.add_argument(
    "--model", type=str, default="new_gate", help="type of model to use"
)
parser.add_argument("--device", type=int, default=0, help="select GPU")
args = parser.parse_args()

# torch.cuda.set_device(args.device)

# Set the random seed manually for reproducibility.
# torch.manual_seed(args.seed)
# if torch.cuda.is_available():
#    if not args.cuda:
#        print("WARNING: You have a CUDA device, so you should probably run with --cuda")
#    else:
#        torch.cuda.manual_seed(args.seed)

torch.manual_seed(args.seed)
if torch.cuda.is_available():
    if not args.cuda:
        # print("WARNING: You have a CUDA device, so you should probably run with --cuda")
        # print("WARNING: You have a CUDA device, so you should probably run with --cuda")
        mylog("Use CPU")
        mylog("Use CPU")
    else:
        # print("Use GPU")
        mylog("Use GPU")
        torch.cuda.set_device(args.device)

###############################################################################
# Load data
###############################################################################

# corpus = data.Corpus(args.data)
corpus = data.TreeFormatCorpus(args.train_file, args.dev_file, args.test_file)


def batchify(data, bsz):
    # Ad hoc, but necessary....
    if bsz > len(data):
        bsz = len(data)

    # Work out how cleanly we can divide the dataset into bsz parts.
    nbatch = len(data) // bsz
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data = data[0 : nbatch * bsz]

    # Evenly divide the data across the bsz batches.
    def list2batch(x_list):
        maxlen = max([len(x) for x in x_list])
        input = torch.LongTensor(maxlen, bsz).zero_()
        mask = torch.FloatTensor(maxlen, bsz).zero_()
        target = torch.LongTensor(maxlen, bsz).zero_()
        for idx, x in enumerate(x_list):
            input[: len(x), idx] = x
            mask[: len(x) - 1, idx] = 1
            target[: len(x) - 1, idx] = x[1:]
        if args.cuda:
            input = input.cuda()
            mask = mask.cuda()
            target = target.cuda()
        return input, mask, target.view(-1)

    data_batched = []
    for i in range(nbatch):
        batch = data[i * bsz : (i + 1) * bsz]
        input, mask, target = list2batch(batch)
        data_batched.append((input, mask, target))

    return data_batched


# Not used.
# eval_batch_size = 64

train_data = batchify(corpus.train, args.batch_size)
val_data = batchify(corpus.valid, args.batch_size)
# val_data = batchify(corpus.valid, eval_batch_size)

# test data not used in this script.
# test_data = batchify(corpus.test, eval_batch_size)

###############################################################################
# Build the model
###############################################################################

ntokens = len(corpus.dictionary)
model = PRPN(
    ntokens,
    args.emsize,
    args.nhid,
    args.nlayers,
    args.nslots,
    args.nlookback,
    args.resolution,
    args.dropout,
    args.idropout,
    args.rdropout,
    args.tied,
    args.hard,
    args.res,
)
if args.cuda:
    model.cuda()


# criterion = nn.CrossEntropyLoss()
def criterion(input, targets, targets_mask):
    targets_mask = targets_mask.view(-1)
    input = input.view(-1, ntokens)
    input = F.log_softmax(input, dim=-1)
    loss = torch.gather(input, 1, targets[:, None]).view(-1)
    loss = (-loss * targets_mask).sum() / targets_mask.sum()
    return loss


###############################################################################
# Training code
###############################################################################


def repackage_hidden(h):
    """Wraps hidden states in new Variables, to detach them from their history."""
    if type(h) == Variable:
        return Variable(h.data)
    else:
        if isinstance(h, list):
            return [repackage_hidden(v) for v in h]
        else:
            return tuple(repackage_hidden(v) for v in h)


def get_batch(source, i, evaluation=False):
    input, mask, target = source[i]
    input = Variable(input, volatile=evaluation)
    mask = Variable(mask, volatile=evaluation)
    target = Variable(target)
    return input, target, mask


def evaluate(data_source):
    # Turn on evaluation mode which disables dropout.
    model.eval()
    total_loss = 0
    ntokens = len(corpus.dictionary)
    for i in range(len(data_source)):
        data, targets, mask = get_batch(data_source, i, evaluation=True)

        # hidden = model.init_hidden(eval_batch_size)
        hidden = model.init_hidden(args.batch_size)

        output, hidden = model(data, hidden)
        output_flat = output.view(-1, ntokens)
        # total_loss += eval_batch_size * criterion(output_flat, targets, mask).data
        total_loss += args.batch_size * criterion(output_flat, targets, mask).item()

    return total_loss / (len(data_source) * args.batch_size)
    # return total_loss[0] / (len(data_source) * eval_batch_size)


def train():
    # Turn on training mode which enables dropout.
    model.train()
    total_loss = 0
    train_loss = 0
    start_time = time.time()
    ntokens = len(corpus.dictionary)
    for batch in range(len(train_data)):
        data, targets, mask = get_batch(train_data, batch)
        # Starting each batch, we detach the hidden state from how it was previously produced.
        # If we didn't, the model would try backpropagating all the way to start of the dataset.
        hidden = model.init_hidden(args.batch_size)
        optimizer.zero_grad()
        output, _ = model(data, hidden)
        loss = criterion(output.view(-1, ntokens), targets, mask)
        loss.backward()

        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        torch.nn.utils.clip_grad_norm(model.parameters(), args.clip)
        optimizer.step()

        # total_loss += loss.data
        # train_loss += loss.data
        total_loss += loss.item()
        train_loss += loss.item()

        if batch % args.log_interval == 0 and batch > 0:
            # cur_loss = total_loss[0] / args.log_interval
            cur_loss = total_loss / args.log_interval
            elapsed = time.time() - start_time
            # print(
            mylog(
                "| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.2f} | ms/batch {:5.2f} | "
                "loss {:5.2f} | ppl {:8.2f}".format(
                    epoch,
                    batch,
                    len(train_data),
                    lr,
                    elapsed * 1000 / args.log_interval,
                    cur_loss,
                    math.exp(cur_loss),
                )
            )
            total_loss = 0
            start_time = time.time()

    return train_loss / batch
    # return train_loss[0] / batch


# Loop over epochs.
lr = args.lr

best_loss = None
early_stop_count = 0


optimizer = optim.Adam(
    model.parameters(),
    lr=args.lr,
    betas=(0, 0.999),
    eps=1e-9,
    weight_decay=args.weight_decay,
)
scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, "min", 0.5, patience=0)

# At any point you can hit Ctrl + C to break out of training early.
try:
    prev_epoch_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
        epoch_start_time = time.time()
        train_loss = train()
        # test_f1 = test(model, corpus, args.cuda)
        val_loss = evaluate(val_data)

        # print("-" * 89)
        mylog("-" * 89)
        # print(
        mylog(
            "| end of epoch {:3d} | time: {:5.2f}s | train loss {:5.2f} | val loss {:5.2f}".format(
                epoch, (time.time() - epoch_start_time), train_loss, val_loss
            )
        )
        # print("-" * 89)
        mylog("-" * 89)
        # Save the model if the validation loss is the best we've seen so far.
        # if not best_loss or train_loss < best_loss:

        # Don't know why but the model selection is done based on train_loss in the original code.
        if not best_loss or val_loss < best_loss:
            with open(args.save, "wb") as f:
                torch.save(model, f)
            best_loss = val_loss

            early_stop_count = 0
        else:
            early_stop_count += 1

        # Added.
        # Early stopping.
        if early_stop_count >= EARLY_STOP_EPOCHS:
            mylog(
                f"Early stop training at epoch {epoch} since dev loss does not improve for {EARLY_STOP_EPOCHS} epochs"
            )
            break

        # best_loss = train_loss
        scheduler.step(train_loss)

        # Added
        # Check if the train loss converges.
        if abs(prev_epoch_loss - train_loss) < EPS:
            # print(
            mylog(
                "Quit training at epoch {} since loss converges".format(
                    epoch,
                )
            )
            break
        else:
            prev_epoch_loss = train_loss

except KeyboardInterrupt:
    # print("-" * 89)
    # print("Exiting from training early")
    mylog("-" * 89)
    mylog("Exiting from training early")

# Load the best saved model.
# with open(args.save, "rb") as f:
#    model = torch.load(f)

# Run on test data.
# test_f1 = test(model, corpus, args.cuda)
# print("=" * 89)
# print("| End of training | test f1 {:5.2f}".format(test_f1))
# print("=" * 89)
