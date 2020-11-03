from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from gensim.models.word2vec import Word2Vec
from gensim.test.utils import common_texts
from nltk.tokenize import word_tokenize
import pandas as pd

import glob
import logging
logging.basicConfig(level=logging.INFO)
import os

from util import *

SEED = 2021
VDIM = 100 # dimension of vectors


# dataset preparation
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


# initialize document vector with pv-dm
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
        f.write(f"{len(paper_files)} {VDIM}\n")

        for k in model.docvecs.doctags:
            f.write(f"{k} {' '.join([str(i) for i in model[k].tolist()])}\n")
    logging.info("--- Pre-trained doc2vec saved. ---")


# hyperdoc2vec
if not os.path.exists("./models/h-d2v.model"):
    model = Word2Vec(
        size=VDIM,
        workers=1, # to ensure reproducibility
        negative=1000,
        sg=0, # use CBOW model
        cbow_mean=1, # use average vector
        seed=SEED
    )

    model.build_vocab(citation_data)
    model.intersect_word2vec_format("./models/pv-dm.txt", lockf=1.0, binary=False)

    logging.info("--- Training started. ---")
    model.train(
        citation_data,
        epochs=100,
        total_examples=model.corpus_count
    )
    logging.info("--- Training ended. ---")

    model.save("./models/h-d2v.model")


model = Word2Vec.load("./models/h-d2v.model")
#print("IN vectors")
#print(len(model.wv.vectors))
#print(model.wv.vocab)
#print(model.wv.vectors)
print("OUT vectors")
print(len(model.trainables.syn1neg))
print(model.trainables.syn1neg)