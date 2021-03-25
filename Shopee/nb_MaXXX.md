
# Notebook: [Image + Text Baseline](https://www.kaggle.com/finlay/unsupervised-image-text-baseline-in-20min)

- Author: MaXXX
- Score: 0.711
- GPU: Yes

## Uses RAPIDS

Make sure GPU Accelerator is enabled

```
!nvidia-smi
```

### Check CUDA Version

```
!nvcc -V && which nvcc
```

### References

- https://www.kaggle.com/beniel/rapids-cudf-tutorial
- https://www.kaggle.com/beniel/01-introduction-to-rapids


Change this to False before submitting?

```
COMPUTE_CV = True
```

## Pytorch

- [Pretrained Models](https://www.kaggle.com/pvlima/pretrained-pytorch-models)
- https://www.kaggle.com/pvlima/use-pretrained-pytorch-models

Upvote the Pretrained Models then **+ Add data** button Data section.

### Resnet18 Model - A CNN

- Residual Networks.  Handle vanishing gradient
- ResNet follows VGG’s full  3×3  convolutional layer design

#### [Mathworks Description](https://www.mathworks.com/help/deeplearning/ref/resnet18.html)

ResNet-18 is a convolutional neural network that is 18 layers deep. You can load a pretrained version of the network trained on more than a million images from the ImageNet database [1]. The pretrained network can classify images into 1000 object categories, such as keyboard, mouse, pencil, and many animals. As a result, the network has learned rich feature representations for a wide range of images. The network has an image input size of 224-by-224.

#### References

- https://towardsdatascience.com/understanding-and-visualizing-resnets-442284831be8
- https://d2l.ai/chapter_convolutional-modern/resnet.html
- https://pytorch.org/hub/pytorch_vision_resnet/
- https://www.kaggle.com/pytorch/resnet18

## TfidfVectorizer

TF-IDF Term Frequency – Inverse Document

- Term Frequency: This summarizes how often a given word appears within a document.
- Inverse Document Frequency: This downscales words that appear a lot across documents.

TF The number of times a word appears in a document divded by the total number of words in the document.

The log of the number of documents divided by the number of documents that contain the word w. Inverse data frequency determines the weight of rare words across all documents in the corpus

### References

- https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html
- https://machinelearningmastery.com/prepare-text-data-machine-learning-scikit-learn/
- https://www.kaggle.com/suyue715/tfidfvectorizer
- https://towardsdatascience.com/natural-language-processing-feature-engineering-using-tf-idf-e8b9d00e7e76
