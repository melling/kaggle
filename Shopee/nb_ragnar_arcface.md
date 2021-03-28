
# Notebook: [Unsupervised Baseline ArcFace](https://www.kaggle.com/ragnar123/unsupervised-baseline-arcface/)

- Author: ragnar (Kaggle Master)
- Score: 0.720
- GPU: Yes

See also: https://www.kaggle.com/slawekbiel/arcface-explained

[My experience so far in this competition - Ragnar](https://www.kaggle.com/c/shopee-product-matching/discussion/228794)

Read this notebook first to get more detail: https://www.kaggle.com/cdeotte/part-2-rapids-tfidfvectorizer-cv-0-700

## 

# Arcmarginproduct class keras layer

```python
class ArcMarginProduct(tf.keras.layers.Layer):
    '''
    Implements large margin arc distance.

    Reference:
        https://arxiv.org/pdf/1801.07698.pdf
        https://github.com/lyakaap/Landmark2019-1st-and-3rd-Place-Solution/
            blob/master/src/modeling/metric_learning.py
    '''
```

## Get neighbors for image_embeddings (Cell 5)

```python
df, image_predictions = get_neighbors(df, image_embeddings, KNN = 50, image = True)
```

## Get neighbors for text_embeddings

```python
df, text_predictions = get_neighbors(df, text_embeddings, KNN = 50, image = False)
```

Final f1 cv score is 0.9009886685266703