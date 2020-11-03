from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from gensim.models.word2vec import Word2Vec
from nltk.tokenize import word_tokenize
import pandas as pd

import argparse
from distutils.util import strtobool
import glob
import logging
logging.basicConfig(level=logging.INFO)
import os

from util import *

SEED = 2021
VDIM = 100 # dimension of vectors


def prepare_datasets():
    paper_data = []
    paper_files = glob.glob("./data/paper/*") 
    for (uid, filename) in enumerate(paper_files):
        with open(filename, "r") as f:
            paper_data.append(TaggedDocument(words=word_tokenize(f.read()), tags=[simplify_filename(filename)]))

    logging.debug(paper_data[:2])

    citation_data = []
    df = pd.read_csv("./data/citations.tsv", sep="\t")
    for _, row in df.iterrows():
        left_context = word_tokenize(row["left context"])
        left_context = left_context[-50:] # max context window length is 50
        right_context = word_tokenize(row["right context"])
        right_context = right_context[:50] # max context window length is 50

        citation_data.append(
            list(flatten([row["citing"], left_context, row["cited"], right_context]))
        )

    logging.debug(citation_data[:2])

    return paper_data, citation_data


# initialize document vector with pv-dm (retrofitting mode)
def pretraining(paper_data):
    if not os.path.exists("./models/pv-dm.txt"):
        model = Doc2Vec(
            vector_size=VDIM, 
            workers=1, # to ensure reproducibility
            epochs=5,
            dm=1, # use pv-dm
            seed=SEED
        )
        model.build_vocab(paper_data)

        logging.info("--- Pre-training started. ---")
        model.train(
            paper_data,
            epochs=model.epochs,
            total_examples=model.corpus_count
        )
        logging.info("--- Pre-training ended. ---")

        # save pre-trained doc2vec as word2vec format file 
        with open("./models/pv-dm.txt", "w") as f:
            f.write(f"{len(paper_data)} {VDIM}\n")

            for k in model.docvecs.doctags:
                f.write(f"{k} {' '.join([str(i) for i in model[k].tolist()])}\n")
        logging.info("--- Pre-trained doc2vec saved. ---")


# hyperdoc2vec
def training(citation_data, retrofit):
    if ((retrofit and not os.path.exists("./models/h-d2v-retrofit.model"))
        or (not retrofit and not os.path.exists("./models/h-d2v-random.model"))):
        model = Word2Vec(
            size=VDIM,
            workers=1, # to ensure reproducibility
            negative=1000,
            sg=0, # use CBOW model
            cbow_mean=1, # use average vector
            seed=SEED
        )

        model.build_vocab(citation_data)
        if retrofit:
            model.intersect_word2vec_format("./models/pv-dm.txt", lockf=1.0, binary=False)

        logging.info("--- Training started. ---")
        model.train(
            citation_data,
            epochs=100,
            total_examples=model.corpus_count
        )
        logging.info("--- Training ended. ---")

        type_string = "retrofit" if retrofit else "random"
        model.save(f"./models/h-d2v-{type_string}.model")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--retrofit', default=True)
    args = parser.parse_args()

    retrofit = strtobool(args.retrofit)
    paper_data, citation_data = prepare_datasets()

    if retrofit:
        pretraining(paper_data)

    training(citation_data, retrofit)

    type_string = "retrofit" if retrofit else "random"
    model = Word2Vec.load(f"./models/h-d2v-{type_string}.model")
    #print("IN vectors")
    #print(len(model.wv.vectors))
    #print(model.wv.vocab)
    #print(model.wv.vectors)
    print("OUT vectors")
    print(len(model.trainables.syn1neg))
    print(model.trainables.syn1neg)