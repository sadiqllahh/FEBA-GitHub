# F-EBA: Feature-Enhanced Boosting Algorithm

## Layout

```
feba/
├── config.py                     hyperparameters, paths, magic numbers
├── main.py                       end-to-end runner (argparse CLI)
├── preprocessing/
│   ├── contractions.py           contractions dictionary + expansion
│   ├── text_cleaner.py           emojis, mentions, links, special chars
│   └── text_transformer.py       lemma/stem/stop-words/negations
├── captioning/
│   ├── image_downloader.py       Twitter media downloader
│   └── git_captioner.py          HuggingFace GIT-base captioner
├── features/
│   ├── sentiment.py              StanfordCoreNLP wrapper (Java server)
│   ├── word2vec.py               gensim Word2Vec + corpus embedder
│   ├── self_attention.py         numpy self-attention (Q/K/V)
│   ├── bert.py                   HuggingFace BERT embedder
│   └── feature_selection.py      custom decision tree + RFE
├── models/
│   ├── bilstm.py                 BiLSTM weak learner
│   ├── boosting.py               BoostingClassifier (sequential)
│   └── feba.py                   F-EBA (10 weak learners + voting)
├── adversarial.py                FGSM attack + adversarial trainer
├── evaluation.py                 confusion / ROC / PRC / k-fold CV
├── xai.py                        SHAP analyser
├── hyperparameter_tuning.py      norm sweep (batch / instance / layer)
├── generalisation.py             Kyoto dataset evaluation
└── abandoned/
    ├── flickr_captioner.py       early Flickr-8k captioner
    └── glove.py                  GloVe (dropped for poor performance)
```

## Setup

```bash
pip install torch torchvision transformers gensim demoji nltk \
            stanfordcorenlp shap scikit-learn matplotlib pandas pillow
```

For sentiment scoring, download Stanford CoreNLP and start the Java
server on port 9000:

```bash
java -mx4g -cp 'stanford-corenlp-4.5.5/*' \
     edu.stanford.nlp.pipeline.StanfordCoreNLPServer -port 9000 -timeout 15000
```

## Running

Edit `feba/config.py` to point `SHEN_PATH` and `CLPSYCH_PATH` at your
CSVs (columns: `TweetId`, `Tweet`, `Label`, `image`). Then:

```bash
python -m feba.main --skip-images           # text-only pipeline
python -m feba.main --norm-sweep            # +normalisation tuning
python -m feba.main --kyoto data/kyoto.csv  # +Kyoto generalisation
```

## Datasets
CLPsych
Shenetl

`Label` is binary 0/1 in the training CSVs. The Kyoto CSV may contain
the third "candidate" class - it is collapsed to non-depressed (0) by
`generalisation.LABEL_MAP`.
