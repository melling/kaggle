# Notebook: [Before We Start](https://www.kaggle.com/maksymshkliarevskyi/shopee-before-we-start-eda-phash-baseline)

- Author: Maksym Shkliarevskyi
- Score: 0.595
- GPU: No

## Overview

Uses provided image phash to match exact and close matches.

## PHash

The data has 'phash' values for images, which can greatly simplify our work.

Phash algorithm is really simple. It breaks images into fragments (in our case, the shape is 8x8), then analyzes the image structure on luminance (without color information) and simply assigns True or False depending on the value (above or below the mean). In order to analyze the similarity, it is necessary to subtract one phash matrix from another. Similar fragments will receive a null value (True - True = 0, False - False = 0). The closer the sum of all differences is to zero, the more similar the images are.

### match_matrix()

```python
def match_matrix(phash_array):
    """
    A function that checks for matches by phash value.
    Takes phash values as input.
    Output - phash diff matrix (pandas data frame)
    """
```

### Exact Matches

- [11, 12] - image_phash: eab5c295966ac368
- [889, 890, 891] - image_phash: b6c8c835b1b66e0e
- [997,520] - image_phash: 89e1f542325be4e9

### Close Matches

Phash analysis allows you to find matches. It allows you to find not only exact copies but also approximate ones

```python
match = []
for i in range(len(matches)):
    match.append(matches.iloc[i, :][(matches.iloc[i, :] > 0) & 
                                    (matches.iloc[i, :] <= 5)].index.values)
match = pd.Series(match)

match[match.apply(lambda x: len(x) >= 1)]
```

