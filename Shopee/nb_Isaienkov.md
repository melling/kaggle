# Notebook: [Data understanding and analysis](https://www.kaggle.com/isaienkov/shopee-data-understanding-and-analysis)

# Overview

Uses provided image phash to match exact along with exact title match.

Observes products in the same group

# Plot Random Images

```python
def plot_images(images_number):
    
    plot_list = train['image'].sample(n=images_number).tolist()
    size = np.sqrt(images_number)
    if int(size)*int(size) < images_number:
        size = int(size) + 1
        
    plt.figure(figsize=(20, 20))
    
    ind=0
    for image_id in plot_list:
        plt.subplot(size, size, ind + 1)
        image = cv2.imread(os.path.join('../input/shopee-product-matching/train_images/', image_id))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        plt.imshow(image)
        plt.title(image_id, fontsize=12)
        plt.axis("off")
        ind+=1
    plt.show()
    
    plot_images(16)
```

# Plot Images by Group

```python
def plot_images(group):
    
    plot_list = train[train['label_group'] == group]
    plot_list = plot_list['image'].tolist()
    images_number = len(plot_list)
    size = np.sqrt(images_number)
    if int(size)*int(size) < images_number:
        size = int(size) + 1
        
    plt.figure(figsize=(20, 20))
    
    ind=0
    for image_id in plot_list:
        plt.subplot(size, size, ind + 1)
        image = cv2.imread(os.path.join('../input/shopee-product-matching/train_images/', image_id))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        plt.imshow(image)
        plt.title(image_id, fontsize=6)
        plt.axis("off")
        ind+=1
    plt.show()

# Product 3627744656
plot_images(3627744656)
```

### Total number of items in group 3627744656: 51, number of unique titles: 49

```python
sample = train[train['label_group'] == 3627744656]
print('Total number of items in group 3627744656: ' + str(len(sample)) + ', number of unique titles: ' + str(sample['title'].nunique()))
```

# Title Analysis

## Title Length

```python
train['title_len'] = train['title'].str.len()
```

# Submission File
