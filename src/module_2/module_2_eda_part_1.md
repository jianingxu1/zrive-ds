```python
!poetry add pyarrow # to load parquets
!poetry add seaborn
```

    The following packages are already present in the pyproject.toml and will be skipped:
    
      • [36mpyarrow[39m
    
    If you want to update it to the latest compatible version, you can use `poetry update package`.
    If you prefer to upgrade it to the latest available version, you can use `poetry add package@latest`.
    
    Nothing to add.
    The following packages are already present in the pyproject.toml and will be skipped:
    
      • [36mseaborn[39m
    
    If you want to update it to the latest compatible version, you can use `poetry update package`.
    If you prefer to upgrade it to the latest available version, you can use `poetry add package@latest`.
    
    Nothing to add.



```python
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

plt.style.use("ggplot")
```


```python
data_path = "../../data/"
```

# 1. Quick overview of the different datasets
## 1.1 Inventory Dataset


```python
df_inventory = pd.read_parquet(data_path + "inventory.parquet")
df_inventory.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 1733 entries, 0 to 1732
    Data columns (total 6 columns):
     #   Column            Non-Null Count  Dtype  
    ---  ------            --------------  -----  
     0   variant_id        1733 non-null   int64  
     1   price             1733 non-null   float64
     2   compare_at_price  1733 non-null   float64
     3   vendor            1733 non-null   object 
     4   product_type      1733 non-null   object 
     5   tags              1733 non-null   object 
    dtypes: float64(2), int64(1), object(3)
    memory usage: 81.4+ KB



```python
df_inventory.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>variant_id</th>
      <th>price</th>
      <th>compare_at_price</th>
      <th>vendor</th>
      <th>product_type</th>
      <th>tags</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>39587297165444</td>
      <td>3.09</td>
      <td>3.15</td>
      <td>heinz</td>
      <td>condiments-dressings</td>
      <td>[table-sauces, vegan]</td>
    </tr>
    <tr>
      <th>1</th>
      <td>34370361229444</td>
      <td>4.99</td>
      <td>5.50</td>
      <td>whogivesacrap</td>
      <td>toilet-roll-kitchen-roll-tissue</td>
      <td>[b-corp, eco, toilet-rolls]</td>
    </tr>
    <tr>
      <th>2</th>
      <td>34284951863428</td>
      <td>3.69</td>
      <td>3.99</td>
      <td>plenty</td>
      <td>toilet-roll-kitchen-roll-tissue</td>
      <td>[kitchen-roll]</td>
    </tr>
    <tr>
      <th>3</th>
      <td>33667283583108</td>
      <td>1.79</td>
      <td>1.99</td>
      <td>thecheekypanda</td>
      <td>toilet-roll-kitchen-roll-tissue</td>
      <td>[b-corp, cruelty-free, eco, tissue, vegan]</td>
    </tr>
    <tr>
      <th>4</th>
      <td>33803537973380</td>
      <td>1.99</td>
      <td>2.09</td>
      <td>colgate</td>
      <td>dental</td>
      <td>[dental-accessories]</td>
    </tr>
  </tbody>
</table>
</div>



No nulls and the types are correct.


```python
df_inventory.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>variant_id</th>
      <th>price</th>
      <th>compare_at_price</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>1.733000e+03</td>
      <td>1733.000000</td>
      <td>1733.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>3.694880e+13</td>
      <td>6.307351</td>
      <td>7.028881</td>
    </tr>
    <tr>
      <th>std</th>
      <td>2.725674e+12</td>
      <td>7.107218</td>
      <td>7.660542</td>
    </tr>
    <tr>
      <th>min</th>
      <td>3.361529e+13</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>3.427657e+13</td>
      <td>2.490000</td>
      <td>2.850000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>3.927260e+13</td>
      <td>3.990000</td>
      <td>4.490000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>3.948318e+13</td>
      <td>7.490000</td>
      <td>8.210000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>4.016793e+13</td>
      <td>59.990000</td>
      <td>60.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
df_inventory["price"].value_counts()
```




    price
    2.99     86
    1.99     72
    0.00     71
    3.99     66
    4.49     65
             ..
    1.65      1
    0.87      1
    11.35     1
    1.36      1
    15.39     1
    Name: count, Length: 179, dtype: int64



There are a lot of products with price 0.00. Why? Are they gifts? New products
with no assigned price?

## 1.2 Orders Dataset


```python
df_orders = pd.read_parquet(data_path + "orders.parquet")
df_orders.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Index: 8773 entries, 10 to 64538
    Data columns (total 6 columns):
     #   Column          Non-Null Count  Dtype         
    ---  ------          --------------  -----         
     0   id              8773 non-null   int64         
     1   user_id         8773 non-null   object        
     2   created_at      8773 non-null   datetime64[us]
     3   order_date      8773 non-null   datetime64[us]
     4   user_order_seq  8773 non-null   int64         
     5   ordered_items   8773 non-null   object        
    dtypes: datetime64[us](2), int64(2), object(2)
    memory usage: 479.8+ KB


No nulls and data types seem correct.\
We do notice that the indexes go from 10 to 64538, and we only got 8773 of\
them, which means that it could be a subset of the original data set.


```python
df_orders.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>user_id</th>
      <th>created_at</th>
      <th>order_date</th>
      <th>user_order_seq</th>
      <th>ordered_items</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>10</th>
      <td>2204073066628</td>
      <td>62e271062eb827e411bd73941178d29b022f5f2de9d37f...</td>
      <td>2020-04-30 14:32:19</td>
      <td>2020-04-30</td>
      <td>1</td>
      <td>[33618849693828, 33618860179588, 3361887404045...</td>
    </tr>
    <tr>
      <th>20</th>
      <td>2204707520644</td>
      <td>bf591c887c46d5d3513142b6a855dd7ffb9cc00697f6f5...</td>
      <td>2020-04-30 17:39:00</td>
      <td>2020-04-30</td>
      <td>1</td>
      <td>[33618835243140, 33618835964036, 3361886244058...</td>
    </tr>
    <tr>
      <th>21</th>
      <td>2204838822020</td>
      <td>329f08c66abb51f8c0b8a9526670da2d94c0c6eef06700...</td>
      <td>2020-04-30 18:12:30</td>
      <td>2020-04-30</td>
      <td>1</td>
      <td>[33618891145348, 33618893570180, 3361889766618...</td>
    </tr>
    <tr>
      <th>34</th>
      <td>2208967852164</td>
      <td>f6451fce7b1c58d0effbe37fcb4e67b718193562766470...</td>
      <td>2020-05-01 19:44:11</td>
      <td>2020-05-01</td>
      <td>1</td>
      <td>[33618830196868, 33618846580868, 3361891234624...</td>
    </tr>
    <tr>
      <th>49</th>
      <td>2215889436804</td>
      <td>68e872ff888303bff58ec56a3a986f77ddebdbe5c279e7...</td>
      <td>2020-05-03 21:56:14</td>
      <td>2020-05-03</td>
      <td>1</td>
      <td>[33667166699652, 33667166699652, 3366717122163...</td>
    </tr>
  </tbody>
</table>
</div>



There is no quantity in ordered_items. This could mean that if an item is bought\
two times, its id has to be added two times.

## 1.3 Regulars Dataset


```python
df_regulars = pd.read_parquet(data_path + "regulars.parquet")
df_regulars.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Index: 18105 entries, 3 to 37720
    Data columns (total 3 columns):
     #   Column      Non-Null Count  Dtype         
    ---  ------      --------------  -----         
     0   user_id     18105 non-null  object        
     1   variant_id  18105 non-null  int64         
     2   created_at  18105 non-null  datetime64[us]
    dtypes: datetime64[us](1), int64(1), object(1)
    memory usage: 565.8+ KB


No nulls and data types seem correct.\
Same as orders, we don't have all the index entries, this could be only a subset\
of the original data set.


```python
df_regulars.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>user_id</th>
      <th>variant_id</th>
      <th>created_at</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>3</th>
      <td>68e872ff888303bff58ec56a3a986f77ddebdbe5c279e7...</td>
      <td>33618848088196</td>
      <td>2020-04-30 15:07:03</td>
    </tr>
    <tr>
      <th>11</th>
      <td>aed88fc0b004270a62ff1fe4b94141f6b1db1496dbb0c0...</td>
      <td>33667178659972</td>
      <td>2020-05-05 23:34:35</td>
    </tr>
    <tr>
      <th>18</th>
      <td>68e872ff888303bff58ec56a3a986f77ddebdbe5c279e7...</td>
      <td>33619009208452</td>
      <td>2020-04-30 15:07:03</td>
    </tr>
    <tr>
      <th>46</th>
      <td>aed88fc0b004270a62ff1fe4b94141f6b1db1496dbb0c0...</td>
      <td>33667305373828</td>
      <td>2020-05-05 23:34:35</td>
    </tr>
    <tr>
      <th>47</th>
      <td>4594e99557113d5a1c5b59bf31b8704aafe5c7bd180b32...</td>
      <td>33667247341700</td>
      <td>2020-05-06 14:42:11</td>
    </tr>
  </tbody>
</table>
</div>



Also, there is no quantity. If a product has to be added to regular two times,
there will be two entries.

## 1.4 Abandoned Carts Dataset


```python
df_abandoned_carts = pd.read_parquet(data_path + "abandoned_carts.parquet")
df_abandoned_carts.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Index: 5457 entries, 0 to 70050
    Data columns (total 4 columns):
     #   Column      Non-Null Count  Dtype         
    ---  ------      --------------  -----         
     0   id          5457 non-null   int64         
     1   user_id     5457 non-null   object        
     2   created_at  5457 non-null   datetime64[us]
     3   variant_id  5457 non-null   object        
    dtypes: datetime64[us](1), int64(1), object(2)
    memory usage: 213.2+ KB


No nulls and data types seem correct.\
The same thing happens with the indexes.


```python
df_abandoned_carts.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>user_id</th>
      <th>created_at</th>
      <th>variant_id</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>12858560217220</td>
      <td>5c4e5953f13ddc3bc9659a3453356155e5efe4739d7a2b...</td>
      <td>2020-05-20 13:53:24</td>
      <td>[33826459287684, 33826457616516, 3366719212762...</td>
    </tr>
    <tr>
      <th>13</th>
      <td>20352449839236</td>
      <td>9d6187545c005d39e44d0456d87790db18611d7c7379bd...</td>
      <td>2021-06-27 05:24:13</td>
      <td>[34415988179076, 34037940158596, 3450282236326...</td>
    </tr>
    <tr>
      <th>45</th>
      <td>20478401413252</td>
      <td>e83fb0273d70c37a2968fee107113698fd4f389c442c0b...</td>
      <td>2021-07-18 08:23:49</td>
      <td>[34543001337988, 34037939372164, 3411360609088...</td>
    </tr>
    <tr>
      <th>50</th>
      <td>20481783103620</td>
      <td>10c42e10e530284b7c7c50f3a23a98726d5747b8128084...</td>
      <td>2021-07-18 21:29:36</td>
      <td>[33667268116612, 34037940224132, 3443605520397...</td>
    </tr>
    <tr>
      <th>52</th>
      <td>20485321687172</td>
      <td>d9989439524b3f6fc4f41686d043f315fb408b954d6153...</td>
      <td>2021-07-19 12:17:05</td>
      <td>[33667268083844, 34284950454404, 33973246886020]</td>
    </tr>
  </tbody>
</table>
</div>



## 1.5 Users Dataset


```python
df_users = pd.read_parquet(data_path + "users.parquet")
df_users.info()
# first_ordered_at and customer_cohort_month data types could be changed
# to datetime
# There are null values on user_nuts1 and all the
# count_ columns
```

    <class 'pandas.core.frame.DataFrame'>
    Index: 4983 entries, 2160 to 3360
    Data columns (total 10 columns):
     #   Column                 Non-Null Count  Dtype  
    ---  ------                 --------------  -----  
     0   user_id                4983 non-null   object 
     1   user_segment           4983 non-null   object 
     2   user_nuts1             4932 non-null   object 
     3   first_ordered_at       4983 non-null   object 
     4   customer_cohort_month  4983 non-null   object 
     5   count_people           325 non-null    float64
     6   count_adults           325 non-null    float64
     7   count_children         325 non-null    float64
     8   count_babies           325 non-null    float64
     9   count_pets             325 non-null    float64
    dtypes: float64(5), object(5)
    memory usage: 428.2+ KB


There are nulls values in `user_nuts1` and all `count_...` columns.\
Also, `first_ordered_at` and `customer_cohort_mont` columns are dates, so their\
datatypes could be changed to datetime.\


```python
df_users.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>user_id</th>
      <th>user_segment</th>
      <th>user_nuts1</th>
      <th>first_ordered_at</th>
      <th>customer_cohort_month</th>
      <th>count_people</th>
      <th>count_adults</th>
      <th>count_children</th>
      <th>count_babies</th>
      <th>count_pets</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2160</th>
      <td>0e823a42e107461379e5b5613b7aa00537a72e1b0eaa7a...</td>
      <td>Top Up</td>
      <td>UKH</td>
      <td>2021-05-08 13:33:49</td>
      <td>2021-05-01 00:00:00</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1123</th>
      <td>15768ced9bed648f745a7aa566a8895f7a73b9a47c1d4f...</td>
      <td>Top Up</td>
      <td>UKJ</td>
      <td>2021-11-17 16:30:20</td>
      <td>2021-11-01 00:00:00</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1958</th>
      <td>33e0cb6eacea0775e34adbaa2c1dec16b9d6484e6b9324...</td>
      <td>Top Up</td>
      <td>UKD</td>
      <td>2022-03-09 23:12:25</td>
      <td>2022-03-01 00:00:00</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>675</th>
      <td>57ca7591dc79825df0cecc4836a58e6062454555c86c35...</td>
      <td>Top Up</td>
      <td>UKI</td>
      <td>2021-04-23 16:29:02</td>
      <td>2021-04-01 00:00:00</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4694</th>
      <td>085d8e598139ce6fc9f75d9de97960fa9e1457b409ec00...</td>
      <td>Top Up</td>
      <td>UKJ</td>
      <td>2021-11-02 13:50:06</td>
      <td>2021-11-01 00:00:00</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>



The reason for the null values could be that the users had to fill a\
questionnaire, and that those columns were optional.

# 2. Analysis of the Datasets

1. Restore orders with inventory information and analyse most sold products,
   categories and vendors.
2. User profiling.

## 2.1 Orders analysis
Hypothesis:
- There are some products, categories and vendors that customers like and buy.


```python
df_orders.head(2)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>user_id</th>
      <th>created_at</th>
      <th>order_date</th>
      <th>user_order_seq</th>
      <th>ordered_items</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>10</th>
      <td>2204073066628</td>
      <td>62e271062eb827e411bd73941178d29b022f5f2de9d37f...</td>
      <td>2020-04-30 14:32:19</td>
      <td>2020-04-30</td>
      <td>1</td>
      <td>[33618849693828, 33618860179588, 3361887404045...</td>
    </tr>
    <tr>
      <th>20</th>
      <td>2204707520644</td>
      <td>bf591c887c46d5d3513142b6a855dd7ffb9cc00697f6f5...</td>
      <td>2020-04-30 17:39:00</td>
      <td>2020-04-30</td>
      <td>1</td>
      <td>[33618835243140, 33618835964036, 3361886244058...</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Each row represents an order with all the products inside of it
# Transform it so that each product + order is a row
exploded_orders = df_orders.explode("ordered_items").rename(
    columns={"ordered_items": "variant_id"}
)
exploded_orders.head(2)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>user_id</th>
      <th>created_at</th>
      <th>order_date</th>
      <th>user_order_seq</th>
      <th>variant_id</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>10</th>
      <td>2204073066628</td>
      <td>62e271062eb827e411bd73941178d29b022f5f2de9d37f...</td>
      <td>2020-04-30 14:32:19</td>
      <td>2020-04-30</td>
      <td>1</td>
      <td>33618849693828</td>
    </tr>
    <tr>
      <th>10</th>
      <td>2204073066628</td>
      <td>62e271062eb827e411bd73941178d29b022f5f2de9d37f...</td>
      <td>2020-04-30 14:32:19</td>
      <td>2020-04-30</td>
      <td>1</td>
      <td>33618860179588</td>
    </tr>
  </tbody>
</table>
</div>



Get each product's information for all orders.


```python
order_items = pd.merge(exploded_orders, df_inventory, how="left", on="variant_id")
order_items.head(3)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>user_id</th>
      <th>created_at</th>
      <th>order_date</th>
      <th>user_order_seq</th>
      <th>variant_id</th>
      <th>price</th>
      <th>compare_at_price</th>
      <th>vendor</th>
      <th>product_type</th>
      <th>tags</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2204073066628</td>
      <td>62e271062eb827e411bd73941178d29b022f5f2de9d37f...</td>
      <td>2020-04-30 14:32:19</td>
      <td>2020-04-30</td>
      <td>1</td>
      <td>33618849693828</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2204073066628</td>
      <td>62e271062eb827e411bd73941178d29b022f5f2de9d37f...</td>
      <td>2020-04-30 14:32:19</td>
      <td>2020-04-30</td>
      <td>1</td>
      <td>33618860179588</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2204073066628</td>
      <td>62e271062eb827e411bd73941178d29b022f5f2de9d37f...</td>
      <td>2020-04-30 14:32:19</td>
      <td>2020-04-30</td>
      <td>1</td>
      <td>33618874040452</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>




```python
order_items.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 107958 entries, 0 to 107957
    Data columns (total 11 columns):
     #   Column            Non-Null Count   Dtype         
    ---  ------            --------------   -----         
     0   id                107958 non-null  int64         
     1   user_id           107958 non-null  object        
     2   created_at        107958 non-null  datetime64[us]
     3   order_date        107958 non-null  datetime64[us]
     4   user_order_seq    107958 non-null  int64         
     5   variant_id        107958 non-null  object        
     6   price             92361 non-null   float64       
     7   compare_at_price  92361 non-null   float64       
     8   vendor            92361 non-null   object        
     9   product_type      92361 non-null   object        
     10  tags              92361 non-null   object        
    dtypes: datetime64[us](2), float64(2), int64(2), object(5)
    memory usage: 9.1+ MB


We notice there are lots of nulls in those columns we added to get the product's\
information. This means that not all products in the orders are in the inventory.


```python
print(
    f" Percentage of product items in orders that are not in inventory: {100*order_items.price.isna().sum() / order_items.index.size}%"
)
```

     Percentage of product items in orders that are not in inventory: 14.447285055299282%



```python
order_items_with_price = order_items.dropna()
order_items_with_price.head(3)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>user_id</th>
      <th>created_at</th>
      <th>order_date</th>
      <th>user_order_seq</th>
      <th>variant_id</th>
      <th>price</th>
      <th>compare_at_price</th>
      <th>vendor</th>
      <th>product_type</th>
      <th>tags</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>70</th>
      <td>2215889436804</td>
      <td>68e872ff888303bff58ec56a3a986f77ddebdbe5c279e7...</td>
      <td>2020-05-03 21:56:14</td>
      <td>2020-05-03</td>
      <td>1</td>
      <td>33667238658180</td>
      <td>4.19</td>
      <td>5.10</td>
      <td>listerine</td>
      <td>dental</td>
      <td>[mouthwash]</td>
    </tr>
    <tr>
      <th>71</th>
      <td>2215889436804</td>
      <td>68e872ff888303bff58ec56a3a986f77ddebdbe5c279e7...</td>
      <td>2020-05-03 21:56:14</td>
      <td>2020-05-03</td>
      <td>1</td>
      <td>33667238658180</td>
      <td>4.19</td>
      <td>5.10</td>
      <td>listerine</td>
      <td>dental</td>
      <td>[mouthwash]</td>
    </tr>
    <tr>
      <th>76</th>
      <td>2217346236548</td>
      <td>66a7b6a77952abc3ef3246da56fb148814704a3c2b420c...</td>
      <td>2020-05-04 11:25:26</td>
      <td>2020-05-04</td>
      <td>1</td>
      <td>33667206054020</td>
      <td>17.99</td>
      <td>20.65</td>
      <td>ecover</td>
      <td>delicates-stain-remover</td>
      <td>[cruelty-free, delicates-stain-remover, eco, v...</td>
    </tr>
  </tbody>
</table>
</div>




```python
top_order_items_by_quantity = order_items_with_price["variant_id"].value_counts()
top_order_items_by_quantity.iloc[:50].plot(kind="bar", figsize=(10, 4))
plt.title("Top 50 products by quantity sold")
plt.xlabel("variant_id")
plt.ylabel("quantity")
```




    Text(0, 0.5, 'quantity')




    
![png](module_2_eda_part_1_files/module_2_eda_part_1_39_1.png)
    



```python
top_order_items_by_revenue = (
    order_items_with_price.groupby("variant_id")["price"]
    .sum()
    .reset_index()
    .rename(columns={"price": "revenue"})
    .sort_values(by="revenue", ascending=False)
)
top_order_items_by_revenue.iloc[:50].plot(
    kind="bar", x="variant_id", y="revenue", figsize=(10, 4)
)
plt.title("Top 50 products by revenue")
plt.xlabel("variant_id")
plt.ylabel("revenue")
```




    Text(0, 0.5, 'revenue')




    
![png](module_2_eda_part_1_files/module_2_eda_part_1_40_1.png)
    



```python
merged_inventory = (
    df_inventory.merge(top_order_items_by_revenue, how="left", on="variant_id")
    .merge(top_order_items_by_quantity, how="left", on="variant_id")
    .rename(columns={"count": "units_sold"})
)
```


```python
merged_inventory.sort_values(by="revenue", ascending=False).iloc[:10]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>variant_id</th>
      <th>price</th>
      <th>compare_at_price</th>
      <th>vendor</th>
      <th>product_type</th>
      <th>tags</th>
      <th>revenue</th>
      <th>units_sold</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>63</th>
      <td>34081589887108</td>
      <td>10.79</td>
      <td>11.94</td>
      <td>oatly</td>
      <td>long-life-milk-substitutes</td>
      <td>[oat-milk, vegan]</td>
      <td>48414.73</td>
      <td>4487.0</td>
    </tr>
    <tr>
      <th>18</th>
      <td>33667268083844</td>
      <td>15.99</td>
      <td>19.99</td>
      <td>persil</td>
      <td>washing-powder</td>
      <td>[washing-powder]</td>
      <td>10921.17</td>
      <td>683.0</td>
    </tr>
    <tr>
      <th>109</th>
      <td>34284949766276</td>
      <td>8.49</td>
      <td>9.00</td>
      <td>andrex</td>
      <td>toilet-roll-kitchen-roll-tissue</td>
      <td>[toilet-rolls]</td>
      <td>7114.62</td>
      <td>838.0</td>
    </tr>
    <tr>
      <th>203</th>
      <td>34081589461124</td>
      <td>8.99</td>
      <td>10.14</td>
      <td>oatly</td>
      <td>long-life-milk-substitutes</td>
      <td>[oat-milk, vegan]</td>
      <td>5771.58</td>
      <td>642.0</td>
    </tr>
    <tr>
      <th>141</th>
      <td>33824368033924</td>
      <td>15.99</td>
      <td>19.99</td>
      <td>persil</td>
      <td>washing-powder</td>
      <td>[washing-powder]</td>
      <td>5516.55</td>
      <td>345.0</td>
    </tr>
    <tr>
      <th>300</th>
      <td>34284950519940</td>
      <td>9.99</td>
      <td>12.00</td>
      <td>fairy</td>
      <td>dishwashing</td>
      <td>[dishwasher-tablets]</td>
      <td>5504.49</td>
      <td>551.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>34370361229444</td>
      <td>4.99</td>
      <td>5.50</td>
      <td>whogivesacrap</td>
      <td>toilet-roll-kitchen-roll-tissue</td>
      <td>[b-corp, eco, toilet-rolls]</td>
      <td>4685.61</td>
      <td>939.0</td>
    </tr>
    <tr>
      <th>190</th>
      <td>34543001370756</td>
      <td>9.99</td>
      <td>13.00</td>
      <td>fairy</td>
      <td>washing-capsules</td>
      <td>[discontinue, trade-swap, washing-capsules]</td>
      <td>4675.32</td>
      <td>468.0</td>
    </tr>
    <tr>
      <th>43</th>
      <td>39709997760644</td>
      <td>11.49</td>
      <td>12.23</td>
      <td>cocacola</td>
      <td>soft-drinks-mixers</td>
      <td>[fizzy-drinks, gluten-free, vegan]</td>
      <td>4492.59</td>
      <td>391.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>34284951863428</td>
      <td>3.69</td>
      <td>3.99</td>
      <td>plenty</td>
      <td>toilet-roll-kitchen-roll-tissue</td>
      <td>[kitchen-roll]</td>
      <td>4180.77</td>
      <td>1133.0</td>
    </tr>
  </tbody>
</table>
</div>



Top 10 most revenue earned products


```python
merged_inventory.sort_values(by="units_sold", ascending=False).iloc[:10]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>variant_id</th>
      <th>price</th>
      <th>compare_at_price</th>
      <th>vendor</th>
      <th>product_type</th>
      <th>tags</th>
      <th>revenue</th>
      <th>units_sold</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>63</th>
      <td>34081589887108</td>
      <td>10.79</td>
      <td>11.94</td>
      <td>oatly</td>
      <td>long-life-milk-substitutes</td>
      <td>[oat-milk, vegan]</td>
      <td>48414.73</td>
      <td>4487.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>34284951863428</td>
      <td>3.69</td>
      <td>3.99</td>
      <td>plenty</td>
      <td>toilet-roll-kitchen-roll-tissue</td>
      <td>[kitchen-roll]</td>
      <td>4180.77</td>
      <td>1133.0</td>
    </tr>
    <tr>
      <th>928</th>
      <td>34284950356100</td>
      <td>1.99</td>
      <td>3.00</td>
      <td>fairy</td>
      <td>dishwashing</td>
      <td>[discontinue, swapped, washing-up-liquid]</td>
      <td>1898.46</td>
      <td>954.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>34370361229444</td>
      <td>4.99</td>
      <td>5.50</td>
      <td>whogivesacrap</td>
      <td>toilet-roll-kitchen-roll-tissue</td>
      <td>[b-corp, eco, toilet-rolls]</td>
      <td>4685.61</td>
      <td>939.0</td>
    </tr>
    <tr>
      <th>116</th>
      <td>33826465153156</td>
      <td>1.89</td>
      <td>1.99</td>
      <td>clearspring</td>
      <td>tins-packaged-foods</td>
      <td>[gluten-free, meat-alternatives, vegan]</td>
      <td>1670.76</td>
      <td>884.0</td>
    </tr>
    <tr>
      <th>109</th>
      <td>34284949766276</td>
      <td>8.49</td>
      <td>9.00</td>
      <td>andrex</td>
      <td>toilet-roll-kitchen-roll-tissue</td>
      <td>[toilet-rolls]</td>
      <td>7114.62</td>
      <td>838.0</td>
    </tr>
    <tr>
      <th>18</th>
      <td>33667268083844</td>
      <td>15.99</td>
      <td>19.99</td>
      <td>persil</td>
      <td>washing-powder</td>
      <td>[washing-powder]</td>
      <td>10921.17</td>
      <td>683.0</td>
    </tr>
    <tr>
      <th>86</th>
      <td>34284950454404</td>
      <td>3.99</td>
      <td>7.50</td>
      <td>lenor</td>
      <td>fabric-softener-freshener</td>
      <td>[fabric-softener-freshener]</td>
      <td>2725.17</td>
      <td>683.0</td>
    </tr>
    <tr>
      <th>241</th>
      <td>34037939372164</td>
      <td>4.99</td>
      <td>5.25</td>
      <td>andrex</td>
      <td>toilet-roll-kitchen-roll-tissue</td>
      <td>[toilet-rolls]</td>
      <td>3308.37</td>
      <td>663.0</td>
    </tr>
    <tr>
      <th>203</th>
      <td>34081589461124</td>
      <td>8.99</td>
      <td>10.14</td>
      <td>oatly</td>
      <td>long-life-milk-substitutes</td>
      <td>[oat-milk, vegan]</td>
      <td>5771.58</td>
      <td>642.0</td>
    </tr>
  </tbody>
</table>
</div>



Top 10 most sold products


```python
top_vendors_by_products_sold = order_items_with_price["vendor"].value_counts()
top_vendors_by_products_sold.iloc[:50].plot(kind="bar", figsize=(10, 4))
plt.title("Top 50 vendors by number of products sold")
plt.xlabel("vendor")
plt.ylabel("quantity")
```




    Text(0, 0.5, 'quantity')




    
![png](module_2_eda_part_1_files/module_2_eda_part_1_46_1.png)
    



```python
top_vendors_by_products_revenue = (
    order_items_with_price.groupby("vendor")
    .agg({"price": "sum"})
    .rename(columns={"price": "revenue"})
    .sort_values(by="revenue", ascending=False)
)
top_vendors_by_products_revenue.iloc[:50].plot(kind="bar", figsize=(10, 4))
plt.title("Top 50 vendors by revenue")
plt.xlabel("vendor")
plt.ylabel("quantity")
```




    Text(0, 0.5, 'quantity')




    
![png](module_2_eda_part_1_files/module_2_eda_part_1_47_1.png)
    



```python
order_items.price.plot(kind="density")
plt.title("Number of products sold in a certain price")
plt.show()
```


    
![png](module_2_eda_part_1_files/module_2_eda_part_1_48_0.png)
    


Most of the products that sold are in the 0 to 8 price range. However,\
there are certain alterations like in the 10 price, that could be caused by the\
most sold product which is the oatly milk.


```python
top_product_type_by_quantity = order_items_with_price["product_type"].value_counts()
top_product_type_by_quantity.iloc[:50].plot(kind="bar", figsize=(10, 4))
plt.title("Top 50 product types by quantity sold")
plt.xlabel("product_type")
plt.ylabel("quantity")
```




    Text(0, 0.5, 'quantity')




    
![png](module_2_eda_part_1_files/module_2_eda_part_1_50_1.png)
    



```python
top_product_type_by_revenue = (
    order_items_with_price.groupby("product_type")
    .agg({"price": "sum"})
    .rename(columns={"price": "revenue"})
    .sort_values(by="revenue", ascending=False)
)
top_product_type_by_revenue.iloc[:50].plot(kind="bar", figsize=(10, 4))
plt.title("Top 50 product types by revenue")
plt.xlabel("product_type")
plt.ylabel("revenue")
```




    Text(0, 0.5, 'revenue')




    
![png](module_2_eda_part_1_files/module_2_eda_part_1_51_1.png)
    



```python
def compute_order_composition(df: pd.DataFrame) -> pd.DataFrame:
    total_orders = df["id"].nunique()
    return (
        df.drop_duplicates(subset=["id", "product_type"])
        .groupby("product_type")["id"]
        .nunique()
        .reset_index()
        .rename(columns={"id": "n_orders"})
        .assign(pct_orders=lambda x: 100*x.n_orders / total_orders)
    )


# Orders composition by product types
order_composition_by_product_type = compute_order_composition(
    order_items_with_price
).sort_values(by="pct_orders", ascending=False)
order_composition_by_product_type.plot(kind="bar", x="product_type", y="pct_orders", figsize=(14,4))
plt.title("Percentage of Orders Containing Each Product Type")
plt.ylabel("Percentage of Orders (%)")

```




    Text(0, 0.5, 'Percentage of Orders (%)')




    
![png](module_2_eda_part_1_files/module_2_eda_part_1_52_1.png)
    



```python
def compute_inventory_composition(df: pd.DataFrame) -> pd.DataFrame:
    total_products = df["variant_id"].nunique()
    return (
        df.groupby("product_type")["variant_id"]
        .nunique()
        .reset_index()
        .rename(columns={"variant_id": "n_products"})
        .assign(pct_inventory=lambda x: 100*x.n_products / total_products)
    )


# Inventory composition by product types
inventory_composition_by_product_type = compute_inventory_composition(
    df_inventory
).sort_values(by="pct_inventory", ascending=False)
inventory_composition_by_product_type.plot(kind="bar", x="product_type", y="pct_inventory", figsize=(14,4))
plt.title("Percentage Composition of Inventory by Product Type")
plt.ylabel("Percentage of Inventory (%)")
```




    Text(0, 0.5, 'Percentage of Inventory (%)')




    
![png](module_2_eda_part_1_files/module_2_eda_part_1_53_1.png)
    



```python
order_inventory_rank = (
    order_composition_by_product_type.merge(
        inventory_composition_by_product_type, how="left", on="product_type"
    )
    .assign(order_rank=lambda x: x.pct_orders.rank(ascending=False))
    .assign(inventory_rank=lambda x: x.pct_inventory.rank(ascending=False))
)
order_inventory_rank.head(15)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>product_type</th>
      <th>n_orders</th>
      <th>pct_orders</th>
      <th>n_products</th>
      <th>pct_inventory</th>
      <th>order_rank</th>
      <th>inventory_rank</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>cleaning-products</td>
      <td>3500</td>
      <td>40.110016</td>
      <td>160</td>
      <td>9.232545</td>
      <td>1.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>tins-packaged-foods</td>
      <td>3281</td>
      <td>37.600275</td>
      <td>125</td>
      <td>7.212926</td>
      <td>2.0</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>toilet-roll-kitchen-roll-tissue</td>
      <td>3131</td>
      <td>35.881274</td>
      <td>18</td>
      <td>1.038661</td>
      <td>3.0</td>
      <td>32.5</td>
    </tr>
    <tr>
      <th>3</th>
      <td>long-life-milk-substitutes</td>
      <td>2657</td>
      <td>30.449232</td>
      <td>24</td>
      <td>1.384882</td>
      <td>4.0</td>
      <td>25.5</td>
    </tr>
    <tr>
      <th>4</th>
      <td>dishwashing</td>
      <td>2632</td>
      <td>30.162732</td>
      <td>27</td>
      <td>1.557992</td>
      <td>5.0</td>
      <td>22.0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>snacks-confectionery</td>
      <td>1920</td>
      <td>22.003209</td>
      <td>122</td>
      <td>7.039815</td>
      <td>6.0</td>
      <td>3.0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>cooking-ingredients</td>
      <td>1817</td>
      <td>20.822828</td>
      <td>73</td>
      <td>4.212349</td>
      <td>7.0</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>7</th>
      <td>soft-drinks-mixers</td>
      <td>1793</td>
      <td>20.547788</td>
      <td>48</td>
      <td>2.769763</td>
      <td>8.0</td>
      <td>11.0</td>
    </tr>
    <tr>
      <th>8</th>
      <td>condiments-dressings</td>
      <td>1732</td>
      <td>19.848728</td>
      <td>52</td>
      <td>3.000577</td>
      <td>9.0</td>
      <td>7.0</td>
    </tr>
    <tr>
      <th>9</th>
      <td>cereal</td>
      <td>1653</td>
      <td>18.943388</td>
      <td>51</td>
      <td>2.942874</td>
      <td>10.0</td>
      <td>8.0</td>
    </tr>
    <tr>
      <th>10</th>
      <td>fabric-softener-freshener</td>
      <td>1625</td>
      <td>18.622507</td>
      <td>17</td>
      <td>0.980958</td>
      <td>11.0</td>
      <td>35.0</td>
    </tr>
    <tr>
      <th>11</th>
      <td>dental</td>
      <td>1480</td>
      <td>16.960807</td>
      <td>42</td>
      <td>2.423543</td>
      <td>12.0</td>
      <td>15.5</td>
    </tr>
    <tr>
      <th>12</th>
      <td>cooking-sauces</td>
      <td>1476</td>
      <td>16.914967</td>
      <td>43</td>
      <td>2.481246</td>
      <td>13.0</td>
      <td>14.0</td>
    </tr>
    <tr>
      <th>13</th>
      <td>spreads</td>
      <td>1408</td>
      <td>16.135686</td>
      <td>19</td>
      <td>1.096365</td>
      <td>14.0</td>
      <td>29.5</td>
    </tr>
    <tr>
      <th>14</th>
      <td>pasta-rice-noodles</td>
      <td>1404</td>
      <td>16.089846</td>
      <td>66</td>
      <td>3.808425</td>
      <td>15.0</td>
      <td>5.0</td>
    </tr>
  </tbody>
</table>
</div>



There some product types that are really frequent in orders but that they\
do not represent the same proportion of importance in the inventory.


```python
order_items["user_order_seq"].value_counts().plot(kind="bar")
# plt.title("")
plt.xlabel("User i-th order")
plt.ylabel("Number of orders")
```




    Text(0, 0.5, 'Number of orders')




    
![png](module_2_eda_part_1_files/module_2_eda_part_1_56_1.png)
    


### Insights

1. The clear bestseller of our products is the `oatly milk`. Why? Is it correctly priced?
2. The products that give the most revenue seem to be sold as a `pack`. How profitable\
   are they? Should we add more product packs
3. The most popular vendors have `ecologic` and/or `household` products. Consider\
   adding more ecological brands or products
4. The most sold product types and that also, normally bought in basket orders are:\
   `cleaning-products`, `tins-packaged-foods`, `toilet-roll-kitchen-tissue`,\
   `long-life-milk-substitutes`, `dishwashing`.
5. There is a discrepancy on the percentage of the product_types that appear the most
   in orders and the proportion they represent in our inventory.

   Should we explore increasing the variety of products offered in the
   product_type that are the frequent in orders?

   On the other hand, should we explore decreasing in our inventory those
   product_type that are less popular?

   For example, toilet-roll-kitchen-roll-tissue product_type appears the 3rd
   most in orders, however, in our inventory it only represents the 32nd
   product_type of our inventory.

   Observation: We do not know the quantity they represent in inventory,\
   only the number of different products in that category.


## 2.2 User profiling
Hypothesis:
- There are users that will buy certain product-types depending on if they
  have children, babies and/or pets.
- Users who have bought a baby or pet product, could have one.

What to do:
- Obtain the percentage of users with children, babies and/or pets.


```python
# Obtain percentage of users with children, babies and pets
questionnaire_users = df_users.dropna(subset=["count_people"])
filtered_questionnaire_users = questionnaire_users[questionnaire_users["count_people"] != 0.0]

filter_children = filtered_questionnaire_users["count_children"] > 0.0
num_users_with_children = filter_children.sum()

filter_babies = filtered_questionnaire_users["count_babies"] > 0.0
num_users_with_babies = filter_babies.sum()

filter_pets = filtered_questionnaire_users["count_pets"] > 0.0
num_users_with_pets = filter_pets.sum()

num_users_questionnaire = filtered_questionnaire_users.shape[0]
num_users_with_children_and_pets = (filter_children & filter_pets).sum()

print(f"Total number of users: {len(df_users)}")
print(f"Num. of users that answered the questionnaire: {num_users_questionnaire}")
print(f"Pct of users that answered the questionnaire: {100*num_users_questionnaire/len(df_users):.2f}%")
print(f"Pct of users with children: {100*num_users_with_children/num_users_questionnaire:.2f}%")
print(f"Pct of users with babies: {100*num_users_with_babies/num_users_questionnaire:.2f}%")
print(f"Pct of users with pets: {100*num_users_with_pets/num_users_questionnaire:.2f}%")
print(
    f"Pct of users with both children and pets: {100*num_users_with_children_and_pets/num_users_questionnaire:.2f}%"
)
```

    Total number of users: 4983
    Num. of users that answered the questionnaire: 323
    Pct of users that answered the questionnaire: 6.48%
    Pct of users with children: 40.25%
    Pct of users with babies: 7.12%
    Pct of users with pets: 40.25%
    Pct of users with both children and pets: 22.29%


Because we are only considering the users that answered the questionnaire, the\
results may be biased. It could be that users that have children or pets tend to\
be the ones who are willing to answer the questionnaire.

Let's try to get the same information from the orders. We will consider that\
users who have bought a pet or baby product, possibly have one.


```python

order_items_with_price["product_type"].unique()
```




    array(['dental', 'delicates-stain-remover', 'fabric-softener-freshener',
           'cleaning-products', 'haircare', 'washing-liquid-gel',
           'baby-kids-toiletries', 'toilet-roll-kitchen-roll-tissue',
           'deodorant', 'condiments-dressings', 'food-bags-cling-film-foil',
           'hand-soap-sanitisers', 'dishwashing', 'bin-bags', 'skincare',
           'cooking-sauces', 'spreads', 'snacks-confectionery',
           'pasta-rice-noodles', 'tins-packaged-foods', 'coffee',
           'home-baking', 'cooking-ingredients', 'cereal', 'bath-shower-gel',
           'washing-powder', 'tea', 'soft-drinks-mixers', 'period-care',
           'drying-ironing', 'baby-toddler-food', 'washing-capsules',
           'pet-care', 'cat-food', 'shaving-grooming',
           'long-life-milk-substitutes', 'suncare', 'dog-food',
           'superfoods-supplements', 'biscuits-crackers',
           'nappies-nappy-pants', 'household-sundries', 'water-softener',
           'baby-accessories', 'beer', 'premixed-cocktails', 'maternity',
           'baby-milk-formula', 'medicines-treatments', 'other-hot-drinks',
           'medicine-treatments', 'sexual-health', 'wine', 'cider',
           'spirits-liqueurs', 'adult-incontinence', 'low-no-alcohol',
           'mixed-bundles'], dtype=object)




```python
order_items_with_price.head(3)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>user_id</th>
      <th>created_at</th>
      <th>order_date</th>
      <th>user_order_seq</th>
      <th>variant_id</th>
      <th>price</th>
      <th>compare_at_price</th>
      <th>vendor</th>
      <th>product_type</th>
      <th>tags</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>70</th>
      <td>2215889436804</td>
      <td>68e872ff888303bff58ec56a3a986f77ddebdbe5c279e7...</td>
      <td>2020-05-03 21:56:14</td>
      <td>2020-05-03</td>
      <td>1</td>
      <td>33667238658180</td>
      <td>4.19</td>
      <td>5.10</td>
      <td>listerine</td>
      <td>dental</td>
      <td>[mouthwash]</td>
    </tr>
    <tr>
      <th>71</th>
      <td>2215889436804</td>
      <td>68e872ff888303bff58ec56a3a986f77ddebdbe5c279e7...</td>
      <td>2020-05-03 21:56:14</td>
      <td>2020-05-03</td>
      <td>1</td>
      <td>33667238658180</td>
      <td>4.19</td>
      <td>5.10</td>
      <td>listerine</td>
      <td>dental</td>
      <td>[mouthwash]</td>
    </tr>
    <tr>
      <th>76</th>
      <td>2217346236548</td>
      <td>66a7b6a77952abc3ef3246da56fb148814704a3c2b420c...</td>
      <td>2020-05-04 11:25:26</td>
      <td>2020-05-04</td>
      <td>1</td>
      <td>33667206054020</td>
      <td>17.99</td>
      <td>20.65</td>
      <td>ecover</td>
      <td>delicates-stain-remover</td>
      <td>[cruelty-free, delicates-stain-remover, eco, v...</td>
    </tr>
  </tbody>
</table>
</div>



We will focus on pet and baby products because they are the most specific\
compared to the children ones.


```python
# Product types that are related to pets and babies
pet_product_types = [
    "pet-care",
    "cat-food",
    "dog-food",
]
baby_product_types = [
    "baby-kids-toiletries",
    "baby-toddler-food",
    "nappies-nappy-pants",
    "baby-accessories",
    "maternity",
    "baby-milk-formula",
]

def num_rows_with_common_elements(df: pd.DataFrame, array: list[str]) -> int:
    count = 0
    for index, row in df.iterrows():
        common_elements = set(row['product_type']).intersection(array)
        if common_elements:
            count += 1
    return count

num_users_with_order = order_items_with_price["user_id"].nunique()

users_product_types_in_orders = order_items_with_price.groupby("user_id")["product_type"].unique().reset_index()
num_users_order_baby = num_rows_with_common_elements(users_product_types_in_orders, baby_product_types)
num_users_order_pets = num_rows_with_common_elements(users_product_types_in_orders, pet_product_types)
print(f"Num. of users with atleast an order: {num_users_with_order}")
print(f"Pct of users with babies by orders: {num_users_order_baby*100/num_users_with_order:.2f}%")
print(f"Pct of users with pets by orders: {num_users_order_pets*100/num_users_with_order:.2f}%")
```

    Num. of users with atleast an order: 4948
    Pct of users with babies by orders: 13.66%
    Pct of users with pets by orders: 11.78%



```python
users_with_pets = filtered_questionnaire_users[filter_pets]["user_id"]
filt_users_with_pets_orders = order_items_with_price["user_id"].isin(users_with_pets)
users_with_pets_orders_product_type = order_items_with_price[filt_users_with_pets_orders].groupby("user_id")["product_type"].unique().reset_index()
num_users_with_pets_that_bought_pet_product = num_rows_with_common_elements(users_with_pets_orders_product_type, pet_product_types)


users_with_babies = filtered_questionnaire_users[filter_babies]["user_id"]
filt_users_with_babies_orders = order_items_with_price["user_id"].isin(users_with_babies)
users_with_babies_orders_product_type = order_items_with_price[filt_users_with_babies_orders].groupby("user_id")["product_type"].unique().reset_index()
num_users_with_babies_that_bought_baby_product = num_rows_with_common_elements(users_with_babies_orders_product_type, baby_product_types)

print(f"Pct of users with pets that have bought a pet product: {100*num_users_with_pets_that_bought_pet_product/len(users_with_pets):.2f}%")
print(f"Pct of users with babies that have bought a baby product: {100*num_users_with_babies_that_bought_baby_product/len(users_with_babies):.2f}%")
```

    Pct of users with pets that have bought a pet product: 29.23%
    Pct of users with babies that have bought a baby product: 43.48%


### Insights
- Only 6.48% of the users answer the questionnaire. It is in our interest, that\
  more users answer it, so that we can target them with ads about specific products.\
  How can we increase the number of users that answer it? Providing a discount?
- It is possible that users who have pets are more open to answer the questionnaire.
- 29.23% of users who answered that they have pets have bought a pet product.
- 43.48% of users who answered that they have babies have bought a baby product.

#### User information extracted from questionnaire
- 40.25% of the users who answered the questionnaire have children.
- 7.12% of the users who answered the questionnaire have babies.
- 40.25% of the users who answered the questionnaire have pets.
- To these users, we can think about targeting them with ads about products\
  related to babies and/or pets because we know for sure that they have babies\
  and/or pets.

#### User information extracted from orders
- 13.66% of the users who have made at least an order, have bought a baby product.
- 11.78% of the users who have made at least an order, have bought a pet product.
- We can observe that the percentage of users with pets we got from the\
  questionnaire, could be an overestimation of the real percentage.
