# hyperdoc2vec

An unofficial implementation of [hyperdoc2vec (ACL 2018)](https://www.aclweb.org/anthology/P18-1222/).  

This repo also contains an example of papers and citations (check `/data` folder). Indeed this is a "toy" example and you cannot expect meaningful results from it.

Since the authors did not release not only source codes but datasets, the correctness of this implementation cannot be checked. If you have any doubts or question, please open an issue.

## Requirements

Implementation was carried out by using [gensim](https://radimrehurek.com/gensim/), just like the authors did in the original paper.  
I recommend [poetry](https://github.com/python-poetry/poetry) as python package manager (of course you can take alternative approach though).

For more detail in required packages, see `pyproject.toml`.

## Preparation

After cloning, run the commands bellow.

```
poetry install
poetry run python -c "import nltk; nltk.download('punkt')"
```

## Demo

```
PYTHONHASHSEED=2021 poetry run python main.py
```

`PYTHONHASHSEED` must be included if you want to get the reproducible results.