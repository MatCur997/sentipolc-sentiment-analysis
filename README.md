# Sentiment analysis of Italian tweets (SENTIPOLC 2016)

Course project for the Computational Linguistics exam. Binary sentiment
classification (positive or negative) of Italian tweets from the EVALITA 2016
SENTIPOLC dataset, combining linguistic preprocessing with linear classifiers.

## Why

Bag-of-words and TF-IDF features ignore syntax, so "non mi piace" and "mi
piace" share most of their features, and they are sensitive to the noise of
tweets (mentions, URLs, hashtags, emoji). SENTIPOLC tweets also concentrate on
a few political and media topics, so a classifier can learn names of
politicians or TV shows as shortcuts instead of sentiment (topic bias). The
project adds three pieces of linguistic preprocessing to a classic pipeline to
address these issues: syntactic handling of negation, normalization of
tweet-specific noise and removal of domain terms.

## Pipeline

All steps are in `main.py`.

1. **Data.** EVALITA 2016 SENTIPOLC, downloaded from Hugging Face
   (`evalitahf/sentiment_analysis`). Only positive and negative tweets are
   kept; neutral and mixed ones are removed.
2. **Cleaning** (regular expressions). HTML entities decoded, emoji converted
   to Italian text, dataset placeholders (`<URL>`, `<MENTION_n>`) and retweet
   tags removed, hashtags kept as `htag_...` tokens.
3. **Linguistic analysis** (Stanza: tokenization, POS tagging, lemmatization,
   dependency parsing).
   - Words are replaced by their lemma.
   - Negation is propagated: the syntactic head of a negation word and its
     `obj`, `xcomp` and `conj` dependents get the suffix `_NEG`.
   - NLTK Italian stopwords are removed, except negations and adversatives
     (*ma*, *però*).
   - A hand-built list of domain terms (politicians, parties, TV shows,
     economic topics) is removed to reduce topic bias.
   - Punctuation is removed except `!` and `?`.
4. **Model selection.** Grid search with 5-fold cross-validation on F1-macro
   over four combinations: bag-of-words or TF-IDF, with Multinomial Naive
   Bayes or Logistic Regression.
5. **Test evaluation.** The decision threshold of the best model is moved from
   0.50 to 0.70 to reduce false positives caused by irony.
6. **Error analysis.** Share of ironic tweets (SENTIPOLC `iro` label) among
   false positives compared with true negatives; the false positives are saved
   to `results/`. The script also prints the most influential words and
   hashtags for each class and the weight and distribution of `!` and `?`.

## Results

As reported in the exam paper (`docs/paper.pdf`):

- Best model: TF-IDF on unigrams and bigrams (at most 5,000 features) with
  Logistic Regression (L2 regularization, C = 5.0).
- F1-macro in 5-fold cross-validation: 0.7188.
- Test set (1,050 tweets, threshold 0.70): accuracy 0.72, F1-macro 0.64.
  Recall on the positive class is lower than on the negative one (0.42
  against 0.85).
- The feature analysis shows semantic (*problema* as a predictor of negative
  sentiment), lexical and pragmatic cues (`?` has discriminative power).

## Installation

Python 3.9 or later.

```bash
git clone https://github.com/MatCur997/sentipolc-sentiment-analysis.git
cd sentipolc-sentiment-analysis
python -m venv .venv
source .venv/Scripts/activate   # Windows (Git Bash); Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

```bash
python main.py
```

On the first run the script downloads the dataset from Hugging Face, the
Stanza Italian models and the NLTK stopwords. Stanza uses the GPU when one is
available. The script prints each step of the analysis and writes
`results/falsi_positivi_con_ironia.csv`.

## Repository layout

```
main.py                               full pipeline
results/falsi_positivi_con_ironia.csv false positives on the test set, with the irony label
docs/paper.pdf                        exam paper (Italian, 4 pages)
requirements.txt                      dependencies
```

## Status and known limitations

Completed course project (June 2026).

- The library versions used for the reported results were not recorded, so
  `requirements.txt` is not pinned. The only constraint is
  `scikit-learn<1.10`, because the `penalty` parameter of
  `LogisticRegression` used in the grid is removed in that version.
- The list of domain terms was built by hand on this dataset and does not
  transfer to other topics.
- The feature analysis reads the coefficients of the classifier, so it assumes
  that Logistic Regression wins the grid search, as in the reported run.
- Comments, identifiers and printed output are in Italian.

## License

Code: MIT (see `LICENSE`). The SENTIPOLC dataset is downloaded at run time and
is not covered by this license; `results/` contains tweet texts from it.
