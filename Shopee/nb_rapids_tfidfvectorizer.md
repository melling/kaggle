
# Notebook: [[PART 2] - RAPIDS TfidfVectorizer](https://www.kaggle.com/cdeotte/part-2-rapids-tfidfvectorizer-cv-0-700)

- Author: Chris Deotte (Kaggle Grandmaster)
- Score: 0.720
- GPU: Yes

Automatically change COMPUTE_CV

```python
if len(test)>3: COMPUTE_CV = False
```


## Compute Baseline CV Score

A baseline is to predict all items with the same image_phash as being duplicate. Let's calcuate the CV score for this submission.

## Compute RAPIDS Model CV and Infer Submission

Use image embeddings, text embeddings, and phash to create a better model with better CV.

Note how the variable COMPUTE_CV is only True when we commit this notebook. Right now you are reading a commit notebook, so we see test replaced with train and computed CV score. When we submit this notebook, the variable COMPUTE_CV will be False and the submit notebook will not compute CV. Instead it will load the real test dataset with 70,000 rows and find duplicates in the real test dataset.

## Use Image Embeddings

To prevent memory errors, we will compute image embeddings in chunks. And we will find similar images with RAPIDS cuML KNN in chunks.

## Use Text Embeddings

To prevent memory errors, we will find similar titles in chunks. To faciliate this, we will use cosine similarity between text embeddings instead of KNN.

## Use Phash Feature

We will predict all items with the same phash as duplicates

## Compute CV Score

Simple model scores a high CV of 0.700+

