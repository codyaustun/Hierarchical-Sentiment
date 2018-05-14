import os
import argparse
import pickle as pkl
import itertools
from collections import Counter, namedtuple

import spacy
import pandas as pd
from tqdm import tqdm


Sample = namedtuple('Sample', ['rating', 'review'])


def to_array_comp(doc):
        return [[w.orth_ for w in s] for s in doc.sents]


def custom_pipeline(nlp):
    return (nlp.tagger, nlp.parser, to_array_comp)


def parse(path):
    with open(path, "r") as file:
        for line in file:
            fields = line.strip().split(',')
            assert len(fields) == 3, "Rating, subject, and body columns only"
            rating, subject, body = fields
            rating = float(rating.strip('\"'))
            yield rating, subject, body


def build_tuples(path, batch_size=10_000, n_threads=8):
    nlp = spacy.load('en', create_pipeline=custom_pipeline)
    print("--> Parsing rows in {}".format(path))
    df = pd.read_csv(path, header=None, names=['rating', 'subject', 'body'])
    print("--> {} rows in {}".format(len(df), path))

    print('--> a little more preprocessing')
    assert df['rating'].isnull().sum() == 0
    df.fillna('', inplace=True)
    _, ratings, subjects, bodies = zip(*df.itertuples())

    # Tokenize reviews
    print("--> Tokenizing")
    body_tokens = nlp.pipe(bodies, batch_size=batch_size, n_threads=n_threads)
    subject_tokens = nlp.pipe(subjects,
                              batch_size=batch_size, n_threads=n_threads)
    review_tokens = (subject + body for subject, body in zip(subject_tokens, body_tokens))  # noqa: E501
    review_tokens = tqdm(review_tokens, desc="Reviews", total=len(df))

    data = [Sample(*tuple_) for tuple_ in zip(ratings, review_tokens)]

    # Filter out empty reviews
    pre_filter_len = len(data)
    data = [sample for sample in data if len(sample.review) > 0]
    post_filter_len = len(data)
    print('--> Filter {} reviews with 0 length'
          .format(pre_filter_len - post_filter_len))
    return data


def build_dataset(args):

    print("Building dataset from : {}".format(args.input))

    train_file = os.path.join(args.input, 'train.csv')
    assert os.path.exists(train_file), "Training examples file doesn't exist!"
    test_file = os.path.join(args.input, 'test.csv')
    assert os.path.exists(test_file), "Test examples file doesn't exists!"

    print("-> Building training data")
    train_data = build_tuples(train_file, batch_size=args.batch_size,
                              n_threads=args.n_threads)
    print(train_data[0])

    print("-> Building test data")
    test_data = build_tuples(test_file, batch_size=args.batch_size,
                             n_threads=args.n_threads)
    print(test_data[0])

    train_split = ((example, 0) for example in train_data)
    test_split = ((example, 1) for example in test_data)
    data, splits = list(zip(*itertools.chain(train_split, test_split)))

    count = Counter(splits)

    print("Split distribution is the following:")
    print(count)

    # Use split 0 for train and split 1 for test

    return {
        "data": data,
        "splits": splits,
        "rows": ("rating", "review")
    }


def main(args):
    ds = build_dataset(args)
    pkl.dump(ds, open(args.output, "wb"))


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=str)
    parser.add_argument("output", type=str, default="sentences.pkl")
    parser.add_argument("-b", "--batch-size", type=int, default=10_000)
    parser.add_argument("-t", "--n-threads", type=int, default=8)
    args = parser.parse_args()

    main(args)
