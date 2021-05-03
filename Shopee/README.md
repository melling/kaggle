# [Shopee](https://www.kaggle.com/c/shopee-product-matching)

Scoring: [F Score](https://en.wikipedia.org/wiki/F-score)

## Data

https://www.kaggle.com/c/shopee-product-matching/data

### Fields

- posting_id - the ID code for the posting
- image - the image id/md5sum
- image_phash - a perceptual hash of the image
- title - the product description for the posting
- label_group - ID code postings that map to the same product. Training set only.

## Observations

- There are no empty fields
- All fields have duplicates, except posting_id
- Perceptial Hashing is used: https://en.wikipedia.org/wiki/Perceptual_hashing

## Notebooks Studied

| Kaggle | Author | Notes | Score | Runtime |
| ---    | ---  | ---  | ---   | --- |
| [EDA One Stop](https://www.kaggle.com/ishandutta/v5-shopee-indepth-eda-one-stop-for-all-your-needs) |Ishan Dutta (Expert) | [Notes](nb_OneStop.md) |n/a|15m|
|[Before We Start EDA](https://www.kaggle.com/maksymshkliarevskyi/shopee-before-we-start-eda-phash-baseline) | Maksym Shkliarevskyi (Expert)| [Notes](nb_BeforeWeStart.md) | 0.595| |
|[Data understanding and analysis](https://www.kaggle.com/isaienkov/shopee-data-understanding-and-analysis) | Kostiantyn Isaienkov (GM)| [Notes](nb_Isaienkov.md)|0.573||
|[Image + Text Baseline](https://www.kaggle.com/finlay/unsupervised-image-text-baseline-in-20min) | MaXXX (Expert)| [Notes](nb_MaXXX.md)|0.711|755.5|
|[Unsupervised Baseline ArcFace](https://www.kaggle.com/ragnar123/unsupervised-baseline-arcface)| ragnar (Master)|[Notes](nb_ragnar_arcface.md)|0.720|929.3 s|
| [[PART 2] - RAPIDS TfidfVectorizer - [CV 0.700]](https://www.kaggle.com/cdeotte/part-2-rapids-tfidfvectorizer-cv-0-700)| Chris Deotte (GM) | [Notes](nb_rapids_tfidfvectorizer.md)| 0.700|429.3 s|

## Notebooks TODO

- [II. Shopee: Model Training with Pytorch x RAPIDS](https://www.kaggle.com/andradaolteanu/ii-shopee-model-training-with-pytorch-x-rapids)

## Discussions

- [Embeddings, Cosine Distance, and ArcFace Explained](https://www.kaggle.com/c/shopee-product-matching/discussion/226279) - Chris Deotte
- [Techniques Implemented So Far](https://www.kaggle.com/c/shopee-product-matching/discussion/228537) - ragnar
- [How To Compute Competition Metric CV](https://www.kaggle.com/c/shopee-product-matching/discussion/225093) - Chris Deotte

## Misc

- https://analyticsindiamag.com/kaggle-interview-grand-master-christof-henkel/
- https://github.com/psinger/kaggle-landmark-recognition-2020-1st-place
