# Notebook: [OneStop EDA](https://www.kaggle.com/ishandutta/v5-shopee-indepth-eda-one-stop-for-all-your-needs)

No NaN in data

all columns have duplicate values except posting_id

# Special Imports

from PIL import Image

### hvplot

For pandas plotting

https://hvplot.holoviz.org

import hvplot.pandas


### Plotly 

import plotly.express as px


https://plotly.com/python/plotly-express/

## Number of Images in Each Directory

- Number of train images: 32412
- Number of test images:  3

https://www.kaggle.com/ishandutta/v5-shopee-indepth-eda-one-stop-for-all-your-needs

# Useful Utility Functions

```python
def display_multiple_img(images_paths, rows, cols): # shows grid of images
```

Generate WordCloud from title field


# NLP

- clean_title
- clean_title_len
- clean_title_word_count
- clean_title_char_count
- clean_title_avg_word_length

Created Distribution Plots for these new features 


# Label Groups

No. of Unique Label Groups: 11014

# Basic Image Exploration

### 2d histograms

- plt.hist2d()
   - https://www.geeksforgeeks.org/matplotlib-pyplot-hist2d-in-python/
   - https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.hist2d.html


For any image specific classification, clustering, etc. transforms we'll want to collapse spatial dimensions so that we have a matrix of pixels by color channels. **np.reshape()**


Scatter plots are a go to to look for clusters and separatbility in the data, but these are busy and don't reveal density well, so we switch to using **2d histograms** instead

# Eigen Images

Commented code at the moment.  Seems to be doing PCA

```python
from sklearn.decomposition import PCA
```

# Rudimentary Transforms, Edge Detection, Texture

```python
import skimage
from skimage.feature import greycomatrix, greycoprops
from skimage.filters import sobel
```

# GLCM Textures

create texture images so we can characterize each pixel by the texture of its neighborhood.

GLCM is inherently anisotropic but can be averaged so as to be rotation invariant. For more on GLCM, see the tutorial.

https://prism.ucalgary.ca/handle/1880/51900

# HSV Transform

HSV is useful for identifying shadows and illumination, as well as giving us a means to identify similar objects that are distinct by color between scenes (hue), though there's no guarantee the hue will be stable.

# Shadow Detection

We can apply a threshold to the V band now to find dark areas that are probably thresholds. Let's look at the distribution of all values then work interactively to find a good filter value.


