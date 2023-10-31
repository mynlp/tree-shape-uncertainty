import argparse
import json

import data_ptb as data
import nltk
import numpy
import torch
from torch.autograd import Variable


# Test model
def build_tree(depth, sen):
    assert len(depth) == len(sen)

    if len(depth) == 1:
        parse_tree = sen[0]
    else:
        idx_max = numpy.argmax(depth)
        parse_tree = []
        if len(sen[:idx_max]) > 0:
            tree0 = build_tree(depth[:idx_max], sen[:idx_max])
            parse_tree.append(tree0)
        tree1 = sen[idx_max]
        if len(sen[idx_max + 1 :]) > 0:
            tree2 = build_tree(depth[idx_max + 1 :], sen[idx_max + 1 :])
            tree1 = [tree1, tree2]
        if parse_tree == []:
            parse_tree = tree1
        else:
            parse_tree.append(tree1)
    return parse_tree


def parse(model, sen, cuda, dictionary):
    x = numpy.array([dictionary[w] for w in sen])
    input = Variable(torch.LongTensor(x[:, None]))
    if cuda:
        input = input.cuda()

    hidden = model.init_hidden(1)
    _, hidden = model(input, hidden)

    attentions = model.attentions.squeeze().data.cpu().numpy()
    gates = model.gates.squeeze().data.cpu().numpy()

    depth = gates[1:-1]
    sen = sen[1:-1]
    attentions = attentions[1:-1]

    parse_tree = build_tree(depth, sen)

    print(parse_tree)
    # model_out = string(parse_tree)
    model_out = parse_tree

    # model_out, _ = get_brackets(parse_tree)

    return model_out


if __name__ == "__main__":
    numpy.set_printoptions(precision=2, suppress=True, linewidth=5000)

    parser = argparse.ArgumentParser(description="PyTorch PTB Language Model")

    # Model parameters.
    # parser.add_argument(
    #    # "--data", type=str, default="./data/ptb", help="location of the data corpus"
    #    "--data",
    #    type=str,
    #    help="location of the data corpus",
    # )

    # But, only test is used.
    parser.add_argument("--train_file", type=str, help="location of the train corpus")
    parser.add_argument("--dev_file", type=str, help="location of the dev corpus")
    parser.add_argument("--test_file", type=str, help="location of the test corpus")
    parser.add_argument(
        "--checkpoint",
        type=str,
        # default="model/model_UP.pt",
        help="model checkpoint to use",
    )
    parser.add_argument(
        "--input_file",
        type=str,
        # default="demo/input_parse.txt",
        help="input file name in tree format",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        # default="demo/output_parse.txt",
        help="output file name",
    )
    parser.add_argument("--seed", type=int, default=1111, help="random seed")
    parser.add_argument("--cuda", action="store_true", help="use CUDA")
    args = parser.parse_args()

    # Set the random seed manually for reproducibility.
    torch.manual_seed(args.seed)

    # Load model
    with open(args.checkpoint, "rb") as f:
        model = torch.load(f)
        if args.cuda:
            model.cuda()
            torch.cuda.manual_seed(args.seed)
        else:
            model.cpu()

    corpus = data.TreeFormatCorpus(args.train_file, args.dev_file, args.test_file)
    dictionary = corpus.dictionary

    outputs = []
    with open(args.input_file, "r") as f:
        for line in f:
            sen_tree = nltk.Tree.fromstring(line)
            words = ["<s>"] + sen_tree.leaves() + ["</s>"]
            bracket = parse(model, words, args.cuda, dictionary)
            outputs.append(json.dumps(bracket))

    with open(args.output_file, "w") as f:
        f.write("\n".join(outputs))
