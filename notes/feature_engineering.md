## Feature Engineering Course Notes


### I. Baseline Model

- Prepare categorical variables: LabelEncoder()
- Create training, validation, and test splits

### II. Categorical Encodings

One-hot encoding, label encoding

**Count Encoding** - Replace each categorical value with the number of times it appears in a data set. e.g. GB occurs 10 times, replace GB with 10

**Target Encoding** - replace a categorical value with the average value of the target for the value of the feature. e.g. Give CA, replace avg outcome with .28

#### CatBoost Encoding

- Similar to target encoing in that it's based  on the target probability
- target probability is only based on rows before it.

## III. Feature Generation

### Interactions

Create new features by combining categorical variables. e.g. Country=CA, 'Music' => CA_Music

#### Transforming numerical values

- Some models work better when the features are normally distributed, so it might help to transform goals: e.g. sqrt, ln
- Helps to contain outliers
- Tree based models are scale invariant

## IV. Univariate Feature Selection

The simplest fastest moethods are based on univariate statistical tests.

For each feature, measure how strongly the target depends on the feature using a statistical test like Chi-Squared or ANOVA.

scikit-learn module: feature_selection.SelectKBest # 3 different scoring functions: Chi-Squares, ANOVA F-value, mutual information score

F-value measures the linear dependency between featues and target. The score might underestimate the relationship if it's non-linear.

### L1 Regularizatin (LASSO)

As the strength of regularization increases, features which are less important for predicting the target are set to zero.
### Misc Notes

Rare values tend to have similar counts (with values like 1 or 2), so you can classify rare values together at prediction time. Common values with large counts are unlikely to have the same exact count as other values. So, the common/important values get their own grouping.
