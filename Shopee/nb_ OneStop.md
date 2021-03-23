# Notebook: [OneStop EDA](https://www.kaggle.com/ishandutta/v5-shopee-indepth-eda-one-stop-for-all-your-needs)

No NaN in data

all columns have duplicate values except posting_id

# Special Imports

from PIL import Image

### hvplot

https://hvplot.holoviz.org
!pip install hvplot
import hvplot.pandas
For pandas plotting

### Plotly 

import plotly.express as px


## Number of Images in Each Directory

Number of train images: 32412
Number of test images:  3

https://www.kaggle.com/ishandutta/v5-shopee-indepth-eda-one-stop-for-all-your-needs

# Useful Utility Functions

def display_multiple_img(images_paths, rows, cols): # shows grid of images

Generate WordCloud from title field


# NLP

clean_title
clean_title_len
clean_title_word_count
clean_title_char_count
clean_title_avg_word_length

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

