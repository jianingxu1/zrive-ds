# Summary

Goal:
- Find the best linear model that predicts the outcome of a product.

Steps:
- Trained linear models using different combinations of features and parameters.

Result:
- The best performing one is a Ridge Logistic Regression with parameters and\
  training dataset:
  - inverse regularization value c=1e-06
  - features used: "ordered_before", "abandoned_before", "global_popularity",\
    "set_as_regular"
  - threshold: 0.01821537339520326 (can be changed depending on business\
    requirements)

# Beginning of model analysis


```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve, roc_curve, roc_auc_score, auc
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import StandardScaler

plt.style.use("ggplot")
```


```python
data_path = "../../data/feature_frame.csv"
df = pd.read_csv(data_path)
```


```python
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 2880549 entries, 0 to 2880548
    Data columns (total 27 columns):
     #   Column                            Dtype  
    ---  ------                            -----  
     0   variant_id                        int64  
     1   product_type                      object 
     2   order_id                          int64  
     3   user_id                           int64  
     4   created_at                        object 
     5   order_date                        object 
     6   user_order_seq                    int64  
     7   outcome                           float64
     8   ordered_before                    float64
     9   abandoned_before                  float64
     10  active_snoozed                    float64
     11  set_as_regular                    float64
     12  normalised_price                  float64
     13  discount_pct                      float64
     14  vendor                            object 
     15  global_popularity                 float64
     16  count_adults                      float64
     17  count_children                    float64
     18  count_babies                      float64
     19  count_pets                        float64
     20  people_ex_baby                    float64
     21  days_since_purchase_variant_id    float64
     22  avg_days_to_buy_variant_id        float64
     23  std_days_to_buy_variant_id        float64
     24  days_since_purchase_product_type  float64
     25  avg_days_to_buy_product_type      float64
     26  std_days_to_buy_product_type      float64
    dtypes: float64(19), int64(4), object(4)
    memory usage: 593.4+ MB


### Classification of each feature



```python
target_col = "outcome"
info_cols = ["variant_id", "order_id", "user_id", "created_at", "order_date"]

features_cols = [col for col in df.columns if col not in info_cols + [target_col]]
categorical_cols = ["product_type", "vendor"]
binary_cols = ["ordered_before", "abandoned_before", "active_snoozed", "set_as_regular"]
numerical_cols = [
    "normalised_price",
    "discount_pct",
    "global_popularity",
    "count_adults",
    "count_children",
    "count_babies",
    "count_pets",
    "people_ex_baby",
    "days_since_purchase_variant_id",
    "avg_days_to_buy_variant_id",
    "std_days_to_buy_variant_id",
    "days_since_purchase_product_type",
    "avg_days_to_buy_product_type",
    "std_days_to_buy_product_type",
]
```

### Cleaning and Filtering the dataset



```python
def filter_orders_with_minimum_size(
    df: pd.DataFrame, min_size: int = 5
) -> pd.DataFrame:
    """
    Filter orders that have at least 5 products
    """
    filt = df.query("outcome == 1.0").groupby("order_id").size() >= min_size
    order_ids = set(filt[filt].index)
    filtered_df = df[df["order_id"].isin(order_ids)]
    return filtered_df


selected_df = (
    df.pipe(filter_orders_with_minimum_size, min_size=5)
    .assign(created_at=lambda x: pd.to_datetime(x.created_at))
    .assign(order_date=lambda x: pd.to_datetime(x.order_date))
    .sort_values("order_date")
)
```

### Split dataset into train, validation and test


We will do a temporal split of the orders to avoid information leakage between\
splits, as future orders may contain information or dynamics of previous orders.\
Perform following split:

- train order dates <= validation order dates <= test order dates



```python
def get_train_val_test_with_temporal_split_orders(
    df: pd.DataFrame,
    train_proportion: float,
    val_proportion: float,
    test_proportion: float,
):
    unique_orders = df.drop_duplicates(subset=["order_id"])[["order_id", "order_date"]]
    total_orders = len(unique_orders)

    train_count = int(total_orders * train_proportion)
    val_count = int(total_orders * val_proportion)
    test_count = total_orders - train_count - val_count

    train_orders = unique_orders.iloc[:train_count]["order_id"]
    val_orders = unique_orders.iloc[train_count : train_count + val_count]["order_id"]
    test_orders = unique_orders.iloc[train_count + val_count :]["order_id"]

    train_data = df[df["order_id"].isin(train_orders)]
    val_data = df[df["order_id"].isin(val_orders)]
    test_data = df[df["order_id"].isin(test_orders)]
    return train_data, val_data, test_data
```


```python
train_proportion = 0.7
val_proportion = 0.2
test_proportion = 0.1

train_data, val_data, test_data = get_train_val_test_with_temporal_split_orders(
    selected_df, train_proportion, val_proportion, test_proportion
)
```


```python
print("Train start date:", train_data["order_date"].iloc[0])
print("Train end date:", train_data["order_date"].iloc[-1])
print("Validation start date:", val_data["order_date"].iloc[0])
print("Validation end date:", val_data["order_date"].iloc[-1])
print("Test start date:", test_data["order_date"].iloc[0])
print("Test end date:", test_data["order_date"].iloc[-1])
assert train_data["order_date"].iloc[0] <= train_data["order_date"].iloc[-1]
assert train_data["order_date"].iloc[-1] <= val_data["order_date"].iloc[0]
assert val_data["order_date"].iloc[0] <= val_data["order_date"].iloc[-1]
assert val_data["order_date"].iloc[-1] <= test_data["order_date"].iloc[0]
assert test_data["order_date"].iloc[0] <= test_data["order_date"].iloc[-1]
```

    Train start date: 2020-10-05 00:00:00
    Train end date: 2021-02-05 00:00:00
    Validation start date: 2021-02-05 00:00:00
    Validation end date: 2021-02-23 00:00:00
    Test start date: 2021-02-23 00:00:00
    Test end date: 2021-03-03 00:00:00


## Baseline model

We will set a baseline model, which is a super simple predictor only based on\
`global_popularity` feature to predict the target `outcome`. Thus, for the next\
models we train, we will be able to compare how they perform to this one,\
considering them iff they are better than this simple one.



```python
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve, auc


def plot_metrics(
    model_name: str,
    y_pred: pd.Series,
    y_test: pd.Series,
    figure: tuple[plt.Figure, list[plt.Axes]] = None,
):
    fpr, tpr, _ = roc_curve(y_test, y_pred)
    roc_auc = auc(fpr, tpr)

    precision, recall, _ = precision_recall_curve(y_test, y_pred)
    pr_auc = auc(recall, precision)

    if figure is None:
        fig, ax = plt.subplots(1, 2, figsize=(14, 6))
    else:
        fig, ax = figure

    # Plot precision-recall curve
    ax[0].plot(
        recall,
        precision,
        label=f"{model_name}, AUC={pr_auc:0.2f}",
    )
    ax[0].set_xlabel("Recall")
    ax[0].set_ylabel("Precision")
    ax[0].set_title("Precision-Recall Curve")
    ax[0].legend()

    # Plot ROC curve
    ax[1].plot(fpr, tpr, label=f"{model_name}, AUC={roc_auc:0.2f}")
    ax[1].set_xlabel("False Positive Rate")
    ax[1].set_ylabel("True Positive Rate")
    ax[1].set_title("ROC Curve")
    ax[1].legend()

    plt.tight_layout()
```


```python
plot_metrics("Global popularity", val_data["global_popularity"], val_data[target_col])
```


    
![png](module_3_model_files/module_3_model_16_0.png)
    


We will use this model that only uses the `global_popularity` feature for\
prediction as our base model.


## Model training



```python
from typing import Tuple


def target_split(df: pd.DataFrame, target_col: str) -> Tuple[pd.DataFrame, pd.Series]:
    return df.drop(target_col, axis=1), df[target_col]
```


```python
X_train, y_train = target_split(train_data, target_col)
X_val, y_val = target_split(val_data, target_col)
X_test, y_test = target_split(test_data, target_col)
```

First, we will train using only numerical and binary features to see how\
models perform so that we avoid the hassle of handling categorical features.



```python
train_cols = numerical_cols + binary_cols
```

## Ridge Regression

Now, lets try how a linear model such as ridge regression compares to our base\
model, trained using only the numerical and binary features.



```python
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
```

Try for different values of regularisation, and see which one is best.



```python
fig1, ax1 = plt.subplots(nrows=1, ncols=2, figsize=(14, 6))
fig1.suptitle("Train metrics")

fig2, ax2 = plt.subplots(nrows=1, ncols=2, figsize=(14, 6))
fig2.suptitle("Validation metrics")

C_param = [10**k for k in range(-6, 6)]
for i, c in enumerate(C_param):
    lr = make_pipeline(StandardScaler(), LogisticRegression(penalty="l2", C=c))
    lr.fit(X_train[train_cols], y_train)

    y_train_pred = lr.predict_proba(X_train[train_cols])[:, 1]
    plot_metrics(f"Ridge LR with C={c:.0e}", y_train_pred, y_train, (fig1, ax1))

    y_val_pred = lr.predict_proba(X_val[train_cols])[:, 1]
    plot_metrics(f"Ridge LR with C={c:.0e}", y_val_pred, y_val, (fig2, ax2))

plot_metrics(
    "Baseline", val_data["global_popularity"], val_data[target_col], (fig2, ax2)
)

plt.show()
```


    
![png](module_3_model_files/module_3_model_26_0.png)
    



    
![png](module_3_model_files/module_3_model_26_1.png)
    


Insights

- Ridge Logistic Regression models are better than the Baseline model.
- Same performance between train and validation, there is no overfitting.
- More regularization improves AUC in ROC Curve.
- Interpretation of ROC Curve. We know that our dataset is extremely unbalanced,\
  with lots of negatives. If we look at the ROC Curve, even a low FPR such as\
  20-30%, is in our problem, very high.


## Lasso Regression


Lets do the same with lasso regression.



```python
fig1, ax1 = plt.subplots(nrows=1, ncols=2, figsize=(14, 6))
fig1.suptitle("Train metrics")

fig2, ax2 = plt.subplots(nrows=1, ncols=2, figsize=(14, 6))
fig2.suptitle("Validation metrics")

C_param = [10**k for k in range(-6, 6)]
for i, c in enumerate(C_param):
    lr = make_pipeline(
        StandardScaler(), LogisticRegression(penalty="l1", C=c, solver="saga")
    )
    lr.fit(X_train[train_cols], y_train)

    y_train_pred = lr.predict_proba(X_train[train_cols])[:, 1]
    plot_metrics(f"Lasso LR with C={c:.0e}", y_train_pred, y_train, (fig1, ax1))

    y_val_pred = lr.predict_proba(X_val[train_cols])[:, 1]
    plot_metrics(f"Lasso LR with C={c:.0e}", y_val_pred, y_val, (fig2, ax2))

plot_metrics(
    "Baseline", val_data["global_popularity"], val_data[target_col], (fig2, ax2)
)

plt.show()
```


    
![png](module_3_model_files/module_3_model_30_0.png)
    



    
![png](module_3_model_files/module_3_model_30_1.png)
    


Insights

- Ridge and Lasso have similar if not the same performance. So, we will prefer\
  Lasso as it normally uses less features.
- Same insights as Ridge.
- More regularization improves the model, but it also plateaus, in this case\
  when C=1e-3.
- Large regularisation makes it predict randomly. For C=1e-06 and C=1e-05, the\
  PR Curve is wrong, in reality, it only contains two points, either predict all\
  1's or all 0's. Look at ROC Curve for better interpretation.


## Analysis of Coefficient weights

We will use each model's best performing parameters for the analysis.



```python
best_ridge_C = 1e-6
best_lasso_C = 1e-3
lr_ridge = make_pipeline(
    StandardScaler(), LogisticRegression(penalty="l2", C=best_ridge_C)
)
lr_ridge.fit(X_train[train_cols], y_train)

lr_lasso = make_pipeline(
    StandardScaler(), LogisticRegression(penalty="l1", C=best_lasso_C, solver="saga")
)
lr_lasso.fit(X_train[train_cols], y_train)
```




<style>#sk-container-id-1 {
  /* Definition of color scheme common for light and dark mode */
  --sklearn-color-text: black;
  --sklearn-color-line: gray;
  /* Definition of color scheme for unfitted estimators */
  --sklearn-color-unfitted-level-0: #fff5e6;
  --sklearn-color-unfitted-level-1: #f6e4d2;
  --sklearn-color-unfitted-level-2: #ffe0b3;
  --sklearn-color-unfitted-level-3: chocolate;
  /* Definition of color scheme for fitted estimators */
  --sklearn-color-fitted-level-0: #f0f8ff;
  --sklearn-color-fitted-level-1: #d4ebff;
  --sklearn-color-fitted-level-2: #b3dbfd;
  --sklearn-color-fitted-level-3: cornflowerblue;

  /* Specific color for light theme */
  --sklearn-color-text-on-default-background: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, black)));
  --sklearn-color-background: var(--sg-background-color, var(--theme-background, var(--jp-layout-color0, white)));
  --sklearn-color-border-box: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, black)));
  --sklearn-color-icon: #696969;

  @media (prefers-color-scheme: dark) {
    /* Redefinition of color scheme for dark theme */
    --sklearn-color-text-on-default-background: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, white)));
    --sklearn-color-background: var(--sg-background-color, var(--theme-background, var(--jp-layout-color0, #111)));
    --sklearn-color-border-box: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, white)));
    --sklearn-color-icon: #878787;
  }
}

#sk-container-id-1 {
  color: var(--sklearn-color-text);
}

#sk-container-id-1 pre {
  padding: 0;
}

#sk-container-id-1 input.sk-hidden--visually {
  border: 0;
  clip: rect(1px 1px 1px 1px);
  clip: rect(1px, 1px, 1px, 1px);
  height: 1px;
  margin: -1px;
  overflow: hidden;
  padding: 0;
  position: absolute;
  width: 1px;
}

#sk-container-id-1 div.sk-dashed-wrapped {
  border: 1px dashed var(--sklearn-color-line);
  margin: 0 0.4em 0.5em 0.4em;
  box-sizing: border-box;
  padding-bottom: 0.4em;
  background-color: var(--sklearn-color-background);
}

#sk-container-id-1 div.sk-container {
  /* jupyter's `normalize.less` sets `[hidden] { display: none; }`
     but bootstrap.min.css set `[hidden] { display: none !important; }`
     so we also need the `!important` here to be able to override the
     default hidden behavior on the sphinx rendered scikit-learn.org.
     See: https://github.com/scikit-learn/scikit-learn/issues/21755 */
  display: inline-block !important;
  position: relative;
}

#sk-container-id-1 div.sk-text-repr-fallback {
  display: none;
}

div.sk-parallel-item,
div.sk-serial,
div.sk-item {
  /* draw centered vertical line to link estimators */
  background-image: linear-gradient(var(--sklearn-color-text-on-default-background), var(--sklearn-color-text-on-default-background));
  background-size: 2px 100%;
  background-repeat: no-repeat;
  background-position: center center;
}

/* Parallel-specific style estimator block */

#sk-container-id-1 div.sk-parallel-item::after {
  content: "";
  width: 100%;
  border-bottom: 2px solid var(--sklearn-color-text-on-default-background);
  flex-grow: 1;
}

#sk-container-id-1 div.sk-parallel {
  display: flex;
  align-items: stretch;
  justify-content: center;
  background-color: var(--sklearn-color-background);
  position: relative;
}

#sk-container-id-1 div.sk-parallel-item {
  display: flex;
  flex-direction: column;
}

#sk-container-id-1 div.sk-parallel-item:first-child::after {
  align-self: flex-end;
  width: 50%;
}

#sk-container-id-1 div.sk-parallel-item:last-child::after {
  align-self: flex-start;
  width: 50%;
}

#sk-container-id-1 div.sk-parallel-item:only-child::after {
  width: 0;
}

/* Serial-specific style estimator block */

#sk-container-id-1 div.sk-serial {
  display: flex;
  flex-direction: column;
  align-items: center;
  background-color: var(--sklearn-color-background);
  padding-right: 1em;
  padding-left: 1em;
}


/* Toggleable style: style used for estimator/Pipeline/ColumnTransformer box that is
clickable and can be expanded/collapsed.
- Pipeline and ColumnTransformer use this feature and define the default style
- Estimators will overwrite some part of the style using the `sk-estimator` class
*/

/* Pipeline and ColumnTransformer style (default) */

#sk-container-id-1 div.sk-toggleable {
  /* Default theme specific background. It is overwritten whether we have a
  specific estimator or a Pipeline/ColumnTransformer */
  background-color: var(--sklearn-color-background);
}

/* Toggleable label */
#sk-container-id-1 label.sk-toggleable__label {
  cursor: pointer;
  display: block;
  width: 100%;
  margin-bottom: 0;
  padding: 0.5em;
  box-sizing: border-box;
  text-align: center;
}

#sk-container-id-1 label.sk-toggleable__label-arrow:before {
  /* Arrow on the left of the label */
  content: "▸";
  float: left;
  margin-right: 0.25em;
  color: var(--sklearn-color-icon);
}

#sk-container-id-1 label.sk-toggleable__label-arrow:hover:before {
  color: var(--sklearn-color-text);
}

/* Toggleable content - dropdown */

#sk-container-id-1 div.sk-toggleable__content {
  max-height: 0;
  max-width: 0;
  overflow: hidden;
  text-align: left;
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-0);
}

#sk-container-id-1 div.sk-toggleable__content.fitted {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-0);
}

#sk-container-id-1 div.sk-toggleable__content pre {
  margin: 0.2em;
  border-radius: 0.25em;
  color: var(--sklearn-color-text);
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-0);
}

#sk-container-id-1 div.sk-toggleable__content.fitted pre {
  /* unfitted */
  background-color: var(--sklearn-color-fitted-level-0);
}

#sk-container-id-1 input.sk-toggleable__control:checked~div.sk-toggleable__content {
  /* Expand drop-down */
  max-height: 200px;
  max-width: 100%;
  overflow: auto;
}

#sk-container-id-1 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {
  content: "▾";
}

/* Pipeline/ColumnTransformer-specific style */

#sk-container-id-1 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {
  color: var(--sklearn-color-text);
  background-color: var(--sklearn-color-unfitted-level-2);
}

#sk-container-id-1 div.sk-label.fitted input.sk-toggleable__control:checked~label.sk-toggleable__label {
  background-color: var(--sklearn-color-fitted-level-2);
}

/* Estimator-specific style */

/* Colorize estimator box */
#sk-container-id-1 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-2);
}

#sk-container-id-1 div.sk-estimator.fitted input.sk-toggleable__control:checked~label.sk-toggleable__label {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-2);
}

#sk-container-id-1 div.sk-label label.sk-toggleable__label,
#sk-container-id-1 div.sk-label label {
  /* The background is the default theme color */
  color: var(--sklearn-color-text-on-default-background);
}

/* On hover, darken the color of the background */
#sk-container-id-1 div.sk-label:hover label.sk-toggleable__label {
  color: var(--sklearn-color-text);
  background-color: var(--sklearn-color-unfitted-level-2);
}

/* Label box, darken color on hover, fitted */
#sk-container-id-1 div.sk-label.fitted:hover label.sk-toggleable__label.fitted {
  color: var(--sklearn-color-text);
  background-color: var(--sklearn-color-fitted-level-2);
}

/* Estimator label */

#sk-container-id-1 div.sk-label label {
  font-family: monospace;
  font-weight: bold;
  display: inline-block;
  line-height: 1.2em;
}

#sk-container-id-1 div.sk-label-container {
  text-align: center;
}

/* Estimator-specific */
#sk-container-id-1 div.sk-estimator {
  font-family: monospace;
  border: 1px dotted var(--sklearn-color-border-box);
  border-radius: 0.25em;
  box-sizing: border-box;
  margin-bottom: 0.5em;
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-0);
}

#sk-container-id-1 div.sk-estimator.fitted {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-0);
}

/* on hover */
#sk-container-id-1 div.sk-estimator:hover {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-2);
}

#sk-container-id-1 div.sk-estimator.fitted:hover {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-2);
}

/* Specification for estimator info (e.g. "i" and "?") */

/* Common style for "i" and "?" */

.sk-estimator-doc-link,
a:link.sk-estimator-doc-link,
a:visited.sk-estimator-doc-link {
  float: right;
  font-size: smaller;
  line-height: 1em;
  font-family: monospace;
  background-color: var(--sklearn-color-background);
  border-radius: 1em;
  height: 1em;
  width: 1em;
  text-decoration: none !important;
  margin-left: 1ex;
  /* unfitted */
  border: var(--sklearn-color-unfitted-level-1) 1pt solid;
  color: var(--sklearn-color-unfitted-level-1);
}

.sk-estimator-doc-link.fitted,
a:link.sk-estimator-doc-link.fitted,
a:visited.sk-estimator-doc-link.fitted {
  /* fitted */
  border: var(--sklearn-color-fitted-level-1) 1pt solid;
  color: var(--sklearn-color-fitted-level-1);
}

/* On hover */
div.sk-estimator:hover .sk-estimator-doc-link:hover,
.sk-estimator-doc-link:hover,
div.sk-label-container:hover .sk-estimator-doc-link:hover,
.sk-estimator-doc-link:hover {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-3);
  color: var(--sklearn-color-background);
  text-decoration: none;
}

div.sk-estimator.fitted:hover .sk-estimator-doc-link.fitted:hover,
.sk-estimator-doc-link.fitted:hover,
div.sk-label-container:hover .sk-estimator-doc-link.fitted:hover,
.sk-estimator-doc-link.fitted:hover {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-3);
  color: var(--sklearn-color-background);
  text-decoration: none;
}

/* Span, style for the box shown on hovering the info icon */
.sk-estimator-doc-link span {
  display: none;
  z-index: 9999;
  position: relative;
  font-weight: normal;
  right: .2ex;
  padding: .5ex;
  margin: .5ex;
  width: min-content;
  min-width: 20ex;
  max-width: 50ex;
  color: var(--sklearn-color-text);
  box-shadow: 2pt 2pt 4pt #999;
  /* unfitted */
  background: var(--sklearn-color-unfitted-level-0);
  border: .5pt solid var(--sklearn-color-unfitted-level-3);
}

.sk-estimator-doc-link.fitted span {
  /* fitted */
  background: var(--sklearn-color-fitted-level-0);
  border: var(--sklearn-color-fitted-level-3);
}

.sk-estimator-doc-link:hover span {
  display: block;
}

/* "?"-specific style due to the `<a>` HTML tag */

#sk-container-id-1 a.estimator_doc_link {
  float: right;
  font-size: 1rem;
  line-height: 1em;
  font-family: monospace;
  background-color: var(--sklearn-color-background);
  border-radius: 1rem;
  height: 1rem;
  width: 1rem;
  text-decoration: none;
  /* unfitted */
  color: var(--sklearn-color-unfitted-level-1);
  border: var(--sklearn-color-unfitted-level-1) 1pt solid;
}

#sk-container-id-1 a.estimator_doc_link.fitted {
  /* fitted */
  border: var(--sklearn-color-fitted-level-1) 1pt solid;
  color: var(--sklearn-color-fitted-level-1);
}

/* On hover */
#sk-container-id-1 a.estimator_doc_link:hover {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-3);
  color: var(--sklearn-color-background);
  text-decoration: none;
}

#sk-container-id-1 a.estimator_doc_link.fitted:hover {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-3);
}
</style><div id="sk-container-id-1" class="sk-top-container"><div class="sk-text-repr-fallback"><pre>Pipeline(steps=[(&#x27;standardscaler&#x27;, StandardScaler()),
                (&#x27;logisticregression&#x27;,
                 LogisticRegression(C=0.001, penalty=&#x27;l1&#x27;, solver=&#x27;saga&#x27;))])</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class="sk-container" hidden><div class="sk-item sk-dashed-wrapped"><div class="sk-label-container"><div class="sk-label fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-1" type="checkbox" ><label for="sk-estimator-id-1" class="sk-toggleable__label fitted sk-toggleable__label-arrow fitted">&nbsp;&nbsp;Pipeline<a class="sk-estimator-doc-link fitted" rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.4/modules/generated/sklearn.pipeline.Pipeline.html">?<span>Documentation for Pipeline</span></a><span class="sk-estimator-doc-link fitted">i<span>Fitted</span></span></label><div class="sk-toggleable__content fitted"><pre>Pipeline(steps=[(&#x27;standardscaler&#x27;, StandardScaler()),
                (&#x27;logisticregression&#x27;,
                 LogisticRegression(C=0.001, penalty=&#x27;l1&#x27;, solver=&#x27;saga&#x27;))])</pre></div> </div></div><div class="sk-serial"><div class="sk-item"><div class="sk-estimator fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-2" type="checkbox" ><label for="sk-estimator-id-2" class="sk-toggleable__label fitted sk-toggleable__label-arrow fitted">&nbsp;StandardScaler<a class="sk-estimator-doc-link fitted" rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.4/modules/generated/sklearn.preprocessing.StandardScaler.html">?<span>Documentation for StandardScaler</span></a></label><div class="sk-toggleable__content fitted"><pre>StandardScaler()</pre></div> </div></div><div class="sk-item"><div class="sk-estimator fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-3" type="checkbox" ><label for="sk-estimator-id-3" class="sk-toggleable__label fitted sk-toggleable__label-arrow fitted">&nbsp;LogisticRegression<a class="sk-estimator-doc-link fitted" rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.4/modules/generated/sklearn.linear_model.LogisticRegression.html">?<span>Documentation for LogisticRegression</span></a></label><div class="sk-toggleable__content fitted"><pre>LogisticRegression(C=0.001, penalty=&#x27;l1&#x27;, solver=&#x27;saga&#x27;)</pre></div> </div></div></div></div></div></div>




```python
lr_coefficients = pd.DataFrame(
    {
        "columns": X_train[train_cols].columns,
        "lasso": lr_lasso[-1].coef_[0],
        "ridge": lr_ridge[-1].coef_[0],
    }
)
lr_coefficients.sort_values("ridge", ascending=True, inplace=True)
lr_coefficients.plot(
    kind="barh",
    x="columns",
    y=["lasso", "ridge"],
    xlabel="features",
    ylabel="value",
    title="Feature coefficients for lasso and ridge models",
)
```




    <Axes: title={'center': 'Feature coefficients for lasso and ridge models'}, xlabel='features', ylabel='value'>




    
![png](module_3_model_files/module_3_model_34_1.png)
    


Insights:

- There are some features that seem to be the ones that help the most when\
  predicting `outcome`, which are: `ordered_before`, `abandoned_before`, and\
  `global_popularity` and `set_as_regular`  kind of.


## Simplified Lasso and Ridge Regression

Train lasso and ridge regression models that only use the most important features\
seen in the previous section.



```python
train_cols = ["ordered_before", "abandoned_before", "global_popularity"]
lr_lasso_simplified = make_pipeline(
    StandardScaler(), LogisticRegression(penalty="l1", C=best_lasso_C, solver="saga")
)
lr_lasso_simplified.fit(X_train[train_cols], y_train)

lr_ridge_simplified = make_pipeline(
    StandardScaler(), LogisticRegression(penalty="l2", C=best_ridge_C)
)
lr_ridge_simplified.fit(X_train[train_cols], y_train)

train_cols = [
    "ordered_before",
    "abandoned_before",
    "global_popularity",
    "set_as_regular",
]
lr_ridge_mid_simplified = make_pipeline(
    StandardScaler(), LogisticRegression(penalty="l2", C=best_ridge_C)
)
lr_ridge_mid_simplified.fit(X_train[train_cols], y_train)
```




<style>#sk-container-id-2 {
  /* Definition of color scheme common for light and dark mode */
  --sklearn-color-text: black;
  --sklearn-color-line: gray;
  /* Definition of color scheme for unfitted estimators */
  --sklearn-color-unfitted-level-0: #fff5e6;
  --sklearn-color-unfitted-level-1: #f6e4d2;
  --sklearn-color-unfitted-level-2: #ffe0b3;
  --sklearn-color-unfitted-level-3: chocolate;
  /* Definition of color scheme for fitted estimators */
  --sklearn-color-fitted-level-0: #f0f8ff;
  --sklearn-color-fitted-level-1: #d4ebff;
  --sklearn-color-fitted-level-2: #b3dbfd;
  --sklearn-color-fitted-level-3: cornflowerblue;

  /* Specific color for light theme */
  --sklearn-color-text-on-default-background: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, black)));
  --sklearn-color-background: var(--sg-background-color, var(--theme-background, var(--jp-layout-color0, white)));
  --sklearn-color-border-box: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, black)));
  --sklearn-color-icon: #696969;

  @media (prefers-color-scheme: dark) {
    /* Redefinition of color scheme for dark theme */
    --sklearn-color-text-on-default-background: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, white)));
    --sklearn-color-background: var(--sg-background-color, var(--theme-background, var(--jp-layout-color0, #111)));
    --sklearn-color-border-box: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, white)));
    --sklearn-color-icon: #878787;
  }
}

#sk-container-id-2 {
  color: var(--sklearn-color-text);
}

#sk-container-id-2 pre {
  padding: 0;
}

#sk-container-id-2 input.sk-hidden--visually {
  border: 0;
  clip: rect(1px 1px 1px 1px);
  clip: rect(1px, 1px, 1px, 1px);
  height: 1px;
  margin: -1px;
  overflow: hidden;
  padding: 0;
  position: absolute;
  width: 1px;
}

#sk-container-id-2 div.sk-dashed-wrapped {
  border: 1px dashed var(--sklearn-color-line);
  margin: 0 0.4em 0.5em 0.4em;
  box-sizing: border-box;
  padding-bottom: 0.4em;
  background-color: var(--sklearn-color-background);
}

#sk-container-id-2 div.sk-container {
  /* jupyter's `normalize.less` sets `[hidden] { display: none; }`
     but bootstrap.min.css set `[hidden] { display: none !important; }`
     so we also need the `!important` here to be able to override the
     default hidden behavior on the sphinx rendered scikit-learn.org.
     See: https://github.com/scikit-learn/scikit-learn/issues/21755 */
  display: inline-block !important;
  position: relative;
}

#sk-container-id-2 div.sk-text-repr-fallback {
  display: none;
}

div.sk-parallel-item,
div.sk-serial,
div.sk-item {
  /* draw centered vertical line to link estimators */
  background-image: linear-gradient(var(--sklearn-color-text-on-default-background), var(--sklearn-color-text-on-default-background));
  background-size: 2px 100%;
  background-repeat: no-repeat;
  background-position: center center;
}

/* Parallel-specific style estimator block */

#sk-container-id-2 div.sk-parallel-item::after {
  content: "";
  width: 100%;
  border-bottom: 2px solid var(--sklearn-color-text-on-default-background);
  flex-grow: 1;
}

#sk-container-id-2 div.sk-parallel {
  display: flex;
  align-items: stretch;
  justify-content: center;
  background-color: var(--sklearn-color-background);
  position: relative;
}

#sk-container-id-2 div.sk-parallel-item {
  display: flex;
  flex-direction: column;
}

#sk-container-id-2 div.sk-parallel-item:first-child::after {
  align-self: flex-end;
  width: 50%;
}

#sk-container-id-2 div.sk-parallel-item:last-child::after {
  align-self: flex-start;
  width: 50%;
}

#sk-container-id-2 div.sk-parallel-item:only-child::after {
  width: 0;
}

/* Serial-specific style estimator block */

#sk-container-id-2 div.sk-serial {
  display: flex;
  flex-direction: column;
  align-items: center;
  background-color: var(--sklearn-color-background);
  padding-right: 1em;
  padding-left: 1em;
}


/* Toggleable style: style used for estimator/Pipeline/ColumnTransformer box that is
clickable and can be expanded/collapsed.
- Pipeline and ColumnTransformer use this feature and define the default style
- Estimators will overwrite some part of the style using the `sk-estimator` class
*/

/* Pipeline and ColumnTransformer style (default) */

#sk-container-id-2 div.sk-toggleable {
  /* Default theme specific background. It is overwritten whether we have a
  specific estimator or a Pipeline/ColumnTransformer */
  background-color: var(--sklearn-color-background);
}

/* Toggleable label */
#sk-container-id-2 label.sk-toggleable__label {
  cursor: pointer;
  display: block;
  width: 100%;
  margin-bottom: 0;
  padding: 0.5em;
  box-sizing: border-box;
  text-align: center;
}

#sk-container-id-2 label.sk-toggleable__label-arrow:before {
  /* Arrow on the left of the label */
  content: "▸";
  float: left;
  margin-right: 0.25em;
  color: var(--sklearn-color-icon);
}

#sk-container-id-2 label.sk-toggleable__label-arrow:hover:before {
  color: var(--sklearn-color-text);
}

/* Toggleable content - dropdown */

#sk-container-id-2 div.sk-toggleable__content {
  max-height: 0;
  max-width: 0;
  overflow: hidden;
  text-align: left;
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-0);
}

#sk-container-id-2 div.sk-toggleable__content.fitted {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-0);
}

#sk-container-id-2 div.sk-toggleable__content pre {
  margin: 0.2em;
  border-radius: 0.25em;
  color: var(--sklearn-color-text);
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-0);
}

#sk-container-id-2 div.sk-toggleable__content.fitted pre {
  /* unfitted */
  background-color: var(--sklearn-color-fitted-level-0);
}

#sk-container-id-2 input.sk-toggleable__control:checked~div.sk-toggleable__content {
  /* Expand drop-down */
  max-height: 200px;
  max-width: 100%;
  overflow: auto;
}

#sk-container-id-2 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {
  content: "▾";
}

/* Pipeline/ColumnTransformer-specific style */

#sk-container-id-2 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {
  color: var(--sklearn-color-text);
  background-color: var(--sklearn-color-unfitted-level-2);
}

#sk-container-id-2 div.sk-label.fitted input.sk-toggleable__control:checked~label.sk-toggleable__label {
  background-color: var(--sklearn-color-fitted-level-2);
}

/* Estimator-specific style */

/* Colorize estimator box */
#sk-container-id-2 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-2);
}

#sk-container-id-2 div.sk-estimator.fitted input.sk-toggleable__control:checked~label.sk-toggleable__label {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-2);
}

#sk-container-id-2 div.sk-label label.sk-toggleable__label,
#sk-container-id-2 div.sk-label label {
  /* The background is the default theme color */
  color: var(--sklearn-color-text-on-default-background);
}

/* On hover, darken the color of the background */
#sk-container-id-2 div.sk-label:hover label.sk-toggleable__label {
  color: var(--sklearn-color-text);
  background-color: var(--sklearn-color-unfitted-level-2);
}

/* Label box, darken color on hover, fitted */
#sk-container-id-2 div.sk-label.fitted:hover label.sk-toggleable__label.fitted {
  color: var(--sklearn-color-text);
  background-color: var(--sklearn-color-fitted-level-2);
}

/* Estimator label */

#sk-container-id-2 div.sk-label label {
  font-family: monospace;
  font-weight: bold;
  display: inline-block;
  line-height: 1.2em;
}

#sk-container-id-2 div.sk-label-container {
  text-align: center;
}

/* Estimator-specific */
#sk-container-id-2 div.sk-estimator {
  font-family: monospace;
  border: 1px dotted var(--sklearn-color-border-box);
  border-radius: 0.25em;
  box-sizing: border-box;
  margin-bottom: 0.5em;
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-0);
}

#sk-container-id-2 div.sk-estimator.fitted {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-0);
}

/* on hover */
#sk-container-id-2 div.sk-estimator:hover {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-2);
}

#sk-container-id-2 div.sk-estimator.fitted:hover {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-2);
}

/* Specification for estimator info (e.g. "i" and "?") */

/* Common style for "i" and "?" */

.sk-estimator-doc-link,
a:link.sk-estimator-doc-link,
a:visited.sk-estimator-doc-link {
  float: right;
  font-size: smaller;
  line-height: 1em;
  font-family: monospace;
  background-color: var(--sklearn-color-background);
  border-radius: 1em;
  height: 1em;
  width: 1em;
  text-decoration: none !important;
  margin-left: 1ex;
  /* unfitted */
  border: var(--sklearn-color-unfitted-level-1) 1pt solid;
  color: var(--sklearn-color-unfitted-level-1);
}

.sk-estimator-doc-link.fitted,
a:link.sk-estimator-doc-link.fitted,
a:visited.sk-estimator-doc-link.fitted {
  /* fitted */
  border: var(--sklearn-color-fitted-level-1) 1pt solid;
  color: var(--sklearn-color-fitted-level-1);
}

/* On hover */
div.sk-estimator:hover .sk-estimator-doc-link:hover,
.sk-estimator-doc-link:hover,
div.sk-label-container:hover .sk-estimator-doc-link:hover,
.sk-estimator-doc-link:hover {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-3);
  color: var(--sklearn-color-background);
  text-decoration: none;
}

div.sk-estimator.fitted:hover .sk-estimator-doc-link.fitted:hover,
.sk-estimator-doc-link.fitted:hover,
div.sk-label-container:hover .sk-estimator-doc-link.fitted:hover,
.sk-estimator-doc-link.fitted:hover {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-3);
  color: var(--sklearn-color-background);
  text-decoration: none;
}

/* Span, style for the box shown on hovering the info icon */
.sk-estimator-doc-link span {
  display: none;
  z-index: 9999;
  position: relative;
  font-weight: normal;
  right: .2ex;
  padding: .5ex;
  margin: .5ex;
  width: min-content;
  min-width: 20ex;
  max-width: 50ex;
  color: var(--sklearn-color-text);
  box-shadow: 2pt 2pt 4pt #999;
  /* unfitted */
  background: var(--sklearn-color-unfitted-level-0);
  border: .5pt solid var(--sklearn-color-unfitted-level-3);
}

.sk-estimator-doc-link.fitted span {
  /* fitted */
  background: var(--sklearn-color-fitted-level-0);
  border: var(--sklearn-color-fitted-level-3);
}

.sk-estimator-doc-link:hover span {
  display: block;
}

/* "?"-specific style due to the `<a>` HTML tag */

#sk-container-id-2 a.estimator_doc_link {
  float: right;
  font-size: 1rem;
  line-height: 1em;
  font-family: monospace;
  background-color: var(--sklearn-color-background);
  border-radius: 1rem;
  height: 1rem;
  width: 1rem;
  text-decoration: none;
  /* unfitted */
  color: var(--sklearn-color-unfitted-level-1);
  border: var(--sklearn-color-unfitted-level-1) 1pt solid;
}

#sk-container-id-2 a.estimator_doc_link.fitted {
  /* fitted */
  border: var(--sklearn-color-fitted-level-1) 1pt solid;
  color: var(--sklearn-color-fitted-level-1);
}

/* On hover */
#sk-container-id-2 a.estimator_doc_link:hover {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-3);
  color: var(--sklearn-color-background);
  text-decoration: none;
}

#sk-container-id-2 a.estimator_doc_link.fitted:hover {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-3);
}
</style><div id="sk-container-id-2" class="sk-top-container"><div class="sk-text-repr-fallback"><pre>Pipeline(steps=[(&#x27;standardscaler&#x27;, StandardScaler()),
                (&#x27;logisticregression&#x27;, LogisticRegression(C=1e-06))])</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class="sk-container" hidden><div class="sk-item sk-dashed-wrapped"><div class="sk-label-container"><div class="sk-label fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-4" type="checkbox" ><label for="sk-estimator-id-4" class="sk-toggleable__label fitted sk-toggleable__label-arrow fitted">&nbsp;&nbsp;Pipeline<a class="sk-estimator-doc-link fitted" rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.4/modules/generated/sklearn.pipeline.Pipeline.html">?<span>Documentation for Pipeline</span></a><span class="sk-estimator-doc-link fitted">i<span>Fitted</span></span></label><div class="sk-toggleable__content fitted"><pre>Pipeline(steps=[(&#x27;standardscaler&#x27;, StandardScaler()),
                (&#x27;logisticregression&#x27;, LogisticRegression(C=1e-06))])</pre></div> </div></div><div class="sk-serial"><div class="sk-item"><div class="sk-estimator fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-5" type="checkbox" ><label for="sk-estimator-id-5" class="sk-toggleable__label fitted sk-toggleable__label-arrow fitted">&nbsp;StandardScaler<a class="sk-estimator-doc-link fitted" rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.4/modules/generated/sklearn.preprocessing.StandardScaler.html">?<span>Documentation for StandardScaler</span></a></label><div class="sk-toggleable__content fitted"><pre>StandardScaler()</pre></div> </div></div><div class="sk-item"><div class="sk-estimator fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-6" type="checkbox" ><label for="sk-estimator-id-6" class="sk-toggleable__label fitted sk-toggleable__label-arrow fitted">&nbsp;LogisticRegression<a class="sk-estimator-doc-link fitted" rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.4/modules/generated/sklearn.linear_model.LogisticRegression.html">?<span>Documentation for LogisticRegression</span></a></label><div class="sk-toggleable__content fitted"><pre>LogisticRegression(C=1e-06)</pre></div> </div></div></div></div></div></div>




```python
from dataclasses import dataclass


@dataclass
class LinearModel:
    model_name: str
    lr: LogisticRegression
    train_cols: list[str]
    C: float
```


```python
fig1, ax1 = plt.subplots(nrows=1, ncols=2, figsize=(14, 6))
fig1.suptitle("Train metrics")

fig2, ax2 = plt.subplots(nrows=1, ncols=2, figsize=(14, 6))
fig2.suptitle("Validation metrics")

models_to_compare = [
    LinearModel(
        model_name="lasso_simplified",
        lr=lr_lasso_simplified,
        train_cols=["ordered_before", "abandoned_before", "global_popularity"],
        C=best_lasso_C,
    ),
    LinearModel(
        model_name="lasso",
        lr=lr_lasso,
        train_cols=numerical_cols + binary_cols,
        C=best_lasso_C,
    ),
    LinearModel(
        model_name="ridge_simplified",
        lr=lr_ridge_simplified,
        train_cols=["ordered_before", "abandoned_before", "global_popularity"],
        C=best_ridge_C,
    ),
    LinearModel(
        model_name="ridge_mid_simplified",
        lr=lr_ridge_mid_simplified,
        train_cols=[
            "ordered_before",
            "abandoned_before",
            "global_popularity",
            "set_as_regular",
        ],
        C=best_ridge_C,
    ),
    LinearModel(
        model_name="ridge",
        lr=lr_ridge,
        train_cols=numerical_cols + binary_cols,
        C=best_ridge_C,
    ),
]

for model in models_to_compare:
    y_train_pred = model.lr.predict_proba(X_train[model.train_cols])[:, 1]
    plot_metrics(
        f"{model.model_name} with C={model.C:.0e}", y_train_pred, y_train, (fig1, ax1)
    )

    y_val_pred = model.lr.predict_proba(X_val[model.train_cols])[:, 1]
    plot_metrics(
        f"{model.model_name} with C={model.C:.0e}", y_val_pred, y_val, (fig2, ax2)
    )

plot_metrics(
    "Baseline", val_data["global_popularity"], val_data[target_col], (fig2, ax2)
)

plt.show()
```


    
![png](module_3_model_files/module_3_model_39_0.png)
    



    
![png](module_3_model_files/module_3_model_39_1.png)
    


Insights

- `lasso_simplified` performs the same as the Lasso one trained with all\
  the features. Thus, we will prefer the simplified version, which will be\
  faster at both training and predicting.
- `ridge_mid_simplified` model performs equal or better than the other ridge\
  versions we will use this model.
- We will use the `ridge_mid_simplified` going on because it seems to be the\
  best performing one and kind of simple.



```python
best_model = LinearModel(
    model_name="ridge_mid_simplified",
    lr=lr_ridge_mid_simplified,
    train_cols=[
        "ordered_before",
        "abandoned_before",
        "global_popularity",
        "set_as_regular",
        # "active_snoozed",
    ],
    C=best_ridge_C,
)
```


```python
y_test_pred = best_model.lr.predict_proba(X_test[best_model.train_cols])[:, 1]
plot_metrics(f"{best_model.model_name} with C={best_model.C:.0e}", y_test_pred, y_test)
```


    
![png](module_3_model_files/module_3_model_42_0.png)
    



```python

from sklearn.metrics import precision_recall_curve, f1_score

precision, recall, thresholds = precision_recall_curve(y_test, y_test_pred)

# Calculate F1 score for each precision-recall pair at different thresholds
f1_scores = []
for i in range(len(thresholds)):
    f1_scores.append(f1_score(y_test, y_test_pred >= thresholds[i]))

max_f1_index = np.argmax(f1_scores)

optimal_threshold = thresholds[max_f1_index]

print("Optimal threshold:", optimal_threshold)
print("Maximum F1 score:", f1_scores[max_f1_index])

```

    Optimal threshold: 0.01821537339520326
    Maximum F1 score: 0.260587942202292


Although the optimal threshold depends on the specific business problem, such\
as the cost of False Positives, etc. because I do not have the information\
regarding these aspects of the business problem, I would have to ask. For now,\
I've decided to use the threshold that has the maximum f1 score which is\
0.01821537339520326.

## Categorical encoding

Analyse if including categorical features improves the model's predictions.\

As seen in the EDA, `product_type` and `vendor` categorical features have high\
cardinality, so encodings that use a lot of columns are not desirable, as they\
increase the dimensionality too much (e.g. one-hot/dummy/binary encoding).\
Also, there is no ordinal relationship between categories, so we discard ordinal\
encoding and label encoding. In this case, I will perform frequency encoding to\
see how it goes, but there are other methods that can work too.

Not enough time to do. To be continued...

