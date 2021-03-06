## Feature Engineering Course Notes


### I. [Baseline Model](https://www.kaggle.com/matleonard/baseline-model)

- Prepare categorical variables: LabelEncoder()
- Create training, validation, and test splits

### II. [Categorical Encodings](https://www.kaggle.com/matleonard/categorical-encodings)

One-hot encoding, label encoding

#### Count Encoding

- Replace each categorical value with the number of times it appears in a data set. e.g. GB occurs 10 times, replace GB with 10

Why is count encoding effective?

Rare values tend to have similar counts (with values like 1 or 2), so you can classify rare values together at prediction time. Common values with large counts are unlikely to have the same exact count as other values. So, the common/important values get their own grouping

```python
import category_encoders as ce
````

#### Target Encoding

- replace a categorical value with the average value of the target for the value of the feature. e.g. Give CA, replace avg outcome with .28

Target encoding attempts to measure the population mean of the target for each level in a categorical feature. This means when there is less data per level, the estimated mean will be further away from the "true" mean, there will be more variance. There is little data per IP address so it's likely that the estimates are much noisier than for the other features. The model will rely heavily on this feature since it is extremely predictive. This causes it to make fewer splits on other features, and those features are fit on just the errors left over accounting for IP address. So, the model will perform very poorly when seeing new IP addresses that weren't in the training data (which is likely most new data). Going forward, we'll leave out the IP feature when trying different encodings.



#### CatBoost Encoding

- Similar to target encoing in that it's based  on the target probability
- target probability is only based on rows before it.

## III. [Feature Generation](https://www.kaggle.com/matleonard/feature-generation)

### Interactions

Create new features by combining categorical variables. e.g. Country=CA, 'Music' => CA_Music

#### Transforming numerical values

- Some models work better when the features are normally distributed, so it might help to transform goals: e.g. sqrt, ln
- Helps to contain outliers
- Tree based models are scale invariant

#### Pandas Time Series

```python
# Numbers events in last 6 hours

def count_past_events(series):
    series = pd.Series(series.index, index=series)
    past_events = series.rolling('6h').count()-1
    return past_events
```

## IV. [Feature Selection](https://www.kaggle.com/matleonard/feature-selection)

Perform feature selection on the train set only, then use the results there to remove features from the validation and test sets.

### Univariate Feature Selection

The simplest fastest methods are based on univariate statistical tests.

For each feature, measure how strongly the target depends on the feature using a statistical test like Chi-Squared or ANOVA.

scikit-learn module: feature_selection.SelectKBest # 3 different scoring functions: Chi-Squares, ANOVA F-value, mutual information score

F-value measures the linear dependency between featues and target. The score might underestimate the relationship if it's non-linear.

"To find the best value of K, you can fit multiple models with increasing values of K, then choose the smallest K with validation score above some threshold or some other criteria. A good way to do this is loop over values of K and record the validation scores for each iteration."

### L1 Regularization (LASSO)

As the strength of regularization increases, features which are less important for predicting the target are set to zero.

"In general, feature selection with L1 regularization is more powerful the univariate tests, but it can also be very slow when you have a lot of data and a lot of features. Univariate tests will be much faster on large datasets, but also will likely perform worse."

```python
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectFromModel

train, valid, _ = get_data_splits(baseline_data)

X, y = train[train.columns.drop("outcome")], train['outcome']

# Set the regularization parameter C=1
logistic = LogisticRegression(C=1, penalty="l1", solver='liblinear', random_state=7).fit(X, y)
model = SelectFromModel(logistic, prefit=True)

X_new = model.transform(X)
X_new
```

### Misc Notes

"Rare values tend to have similar counts (with values like 1 or 2), so you can classify rare values together at prediction time. Common values with large counts are unlikely to have the same exact count as other values. So, the common/important values get their own grouping."

