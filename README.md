# Deplatforming and Misinformation: Efficacy of Twitter's January 2021 User Suspensions

## Introduction

This analysis builds on the study, *Post-January 6th deplatforming reduced the reach of misinformation on Twitter* (McCabe et al. 2024), which examined Twitter's large-scale account suspensions following the January 6th, 2021 insurrection. Noting the significant role misinformation and conspiracy theories played, Twitter and other social media platforms implemented sweeping account suspensions to prevent their spread.

The goal of this project is to replicate and extend the findings of McCabe et al. using an anonymized replication dataset provided by the authors. I aim to further susbtantiate their conclusions to gauge the actual efficacy of Twitter's post-January 6th user suspensions. By doing so, I hope to improve the understanding of how social media platforms can mitigate the spread of misinformation and its impact on public discourse.

### Background and Context

The original research team compiled a pool of over 500,000 active Twitter users that could be cross-verified with a voter registration database, then assembled a dataset containing all of these users' activity between 2019 and 2021. 

Using multiple pre-curated lists of websites known to be sources of misinformation, the researchers focused specifically on tweets and retweets containing links to these websites.

Notes
- Aggregated Twitter data from late 2019 through 2021, focusing on URLs identified as misinformation
- Classified users into categories based on their activity levels and misinformation spread
- Used Difference-in-Differences (DiD) to measure the causal effect of deploatforming on misinformation spread


```python
import pandas as pd
import numpy as np
import seaborn as sns
import re
import json
import warnings
import statsmodels.api as sm
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib_venn as venn
import matplotlib.dates as mdates
import base64
import io, requests
from IPython.display import Image, display
from PIL import Image as im
from datetime import datetime, timedelta, date


warnings.filterwarnings(action="ignore", category=pd.errors.SettingWithCopyWarning)
warnings.filterwarnings(action="ignore", category=FutureWarning)
warnings.filterwarnings(action="ignore", category=RuntimeWarning)
warnings.filterwarnings(action="ignore", category=DeprecationWarning)
```

## Data Preparation and Organization


```python
# load data
mccabe = mccabe = pd.read_csv(
    "mccabe-public-data.csv", on_bad_lines="skip"
).reset_index(names=["ID"])
mccabe["group"] = mccabe["group"]
mccabe["date"] = pd.to_datetime(mccabe["date"], format="%Y-%m-%d")
```

### De-aggregating the data

To protect users' privacy, the replication data is available in an anonymized, aggregated format. The researchers divided the users into overlapping categories based on their activity levels and misinformation spread, then supplied the observed (mostly) daily counts for each group. 

Select Groups:
| Name | Group | Description |
|------|-------|-------------|
| FNS | misinformation sharers | users who share at least 1 URL with misinformation |
| DU | suspended users | users removed between January 6th and January 12th |
| HA | high activity | users who sent at least 3,200 tweets during a six-week collection interval between 2018 and April 2020 |
| MA | medium activity | the most active 500,000 users who didn't meet the high activity threshold |
| LA | low activity | all users who didn't meet the high or medium activity thresholds |
| A | Trump-only followers | non-suspended misinformation sharers who follow Trump but no other deplatformed users |
| B | deplatformed followers | non-suspended misinformation sharers who follow at least one deplatformed user (can include Trump) |
| D | 4+ deplatformed followers | non-suspended misinformation sharers who follow at least four deplatformed users (can include Trump) |
| F | not deplatformed followers | non-suspended misinformation sharers who do not follow any deplatformed users |

I used some probability rules to reorganize the data into a more usable (mutually exclusive) format.


```python
fig, ax = plt.subplots(1, 3, figsize=(18, 10))

v = venn.venn2(subsets=(4, 6, 3), set_labels=(' Group A ', ' Group B '), set_colors=('yellow', 'blue'), alpha=0.5, ax=ax[0])
v.get_label_by_id('10').set_text('')
v.get_label_by_id('01').set_text('')
v.get_label_by_id('11').set_text('')
v.get_patch_by_id('11').set_color('green')

w = venn.venn2(subsets=(4, 6, 3), set_labels=(' Group D ', ' Group B '), set_colors=('red', 'blue'), alpha=0.5, ax=ax[1])
w.get_label_by_id('10').set_text('')
w.get_label_by_id('01').set_text('')
w.get_label_by_id('11').set_text('')
w.get_patch_by_id('11').set_color('purple')

x = venn.venn2(subsets=(4, 4, 0), set_labels=(' Group A ', ' Group D '), set_colors=('yellow', 'red'), alpha=0.5, ax=ax[2])
x.get_label_by_id('10').set_text('')
x.get_label_by_id('01').set_text('')

fig.suptitle('Group Relationships', fontsize=16)
plt.tight_layout()
plt.show()
```


```python
# one entry per group per day
numbers = [col for col in mccabe.columns if col not in ["ID", "date", "stat", "group"]]

mccabe_full = mccabe.groupby(["date", "group", "stat"])[numbers].sum().reset_index()
```

I built several functions to handle the disaggregation of the data. In addition to comparing and subtracting the subsets, I also added empty rows to the dataframes to make sure that the dataframes had the same number of rows. I will be aggregating the data later on, so this will not imapct the inteegrity of the data. It's also important to note that the disaggregation only breaks up the data into mutually exclusive categories, not into individual, user-level observations.


```python
# add empty rows when subset has no activity (assume this is observed in the data)
def add_missing_level(set, set_name):
    """
    Adds missing activity level rows to a DataFrame when a subset has no activity.

    Args:
        set (pandas.DataFrame): The DataFrame to add missing rows to.
        set_name (str): The name of the grouping for the DataFrame.

    Returns:
        pandas.DataFrame: The updated DataFrame with missing rows added.
    """
    if len(set.loc[:, "level"].unique()) < 3:
        level = [
            l for l in ["ha", "ma", "la"] if l not in set.loc[:, "level"].unique()
        ][0]
        empty_row = pd.DataFrame(
            {col: [0 if col != "level" else level] for col in set.columns}
        )
        empty_row["level"] = level
        empty_row["stat"] = "total"
        empty_row["grouping"] = set_name
        if level == "la":
            set = pd.concat([set, empty_row], ignore_index=True)
        if level == "ha":
            set = pd.concat([empty_row, set], ignore_index=True)
    return set


def preprocessing(df_raw, date):
    """
    Preprocesses a raw DataFrame by filtering to a specific date, creating mutually exclusive groups, and processing subsets by activity level.

    Args:
        df_raw (pandas.DataFrame): The raw DataFrame to preprocess.
        date (str): The date to filter the DataFrame to.

    Returns:
        tuple:
            ha (pandas.DataFrame): The 'ha' group DataFrame.
            ma (pandas.DataFrame): The 'ma' group DataFrame.
            la (pandas.DataFrame): The 'la' group DataFrame.
            processed_groups (dict): A dictionary of processed group DataFrames.
            sub_total (pandas.DataFrame): The total subset DataFrame.
            groups (list): A list of group names.
    """
    # data for the given date
    df = df_raw.loc[df_raw["date"] == date].reset_index(drop=True)

    # start with mutually exclusive groups
    ha, ma, la = [
        df.loc[(df["group"] == group) & (df["stat"] == "total")]
        for group in ["ha", "ma", "la"]
    ]

    # subset by activity level
    su = df.loc[df["group"].str.contains(r"\_[hml]a")]
    groupings = (
        su.copy()
        .loc[:, "group"]
        # split the group column into two columns
        .str.split("_", expand=True)
        .rename(columns={0: "grouping", 1: "level"})
        .apply(lambda x: x.str.strip())
    )
    # turn into separate columns
    sub = pd.concat(
        [su.drop(columns=["grouping", "level", "group"], errors="ignore"), groupings],
        axis=1,
    )

    # filter to sum only
    sub_total = sub.copy().loc[sub["stat"] == "total"]

    def process_group(sub_total, group_name):
        """
        Checking for/adding 'ha', 'ma', 'la' levels.
        """
        if "grouping" not in sub_total.columns:
            sub_total["grouping"] = sub_total["group"]
        group = sub_total.loc[sub_total["grouping"] == group_name]
        group = add_missing_level(group, group_name)
        if group_name == "A":
            group["date"] = date
        return group

    # A, D, F, and nfns groups
    groups = ["A", "D", "F", "nfns"]
    processed_groups = {}
    for group in groups:
        result = process_group(sub_total, group)
        if isinstance(result, tuple):
            processed_groups[group] = list(result)
        else:
            processed_groups[group] = [result]

    return ha, ma, la, processed_groups, sub_total, groups


def process_suspended(suspended):
    """Processes the suspended data by creating a common DataFrame with the 'total', 'suspended', and 'level' columns.

    Args:
        suspended (pandas.DataFrame): The suspended DataFrame.

    Returns:
        pandas.DataFrame: The processed suspended DataFrame with the common columns added.
    """

    suspended_common = pd.DataFrame(
        {
            "stat": "total",
            "grouping": "suspended",
            "level": ["ha", "ma", "la"],
        }
    )
    suspended = suspended.reset_index().join(suspended_common, rsuffix="_common")
    return suspended


def pull_B(sub_total, processed_groups):
    """
    Pulls the 'B' group from the sub_total DataFrame and calculates the difference between 'B' and the union of 'A' and 'D' groups.

    Args:
        sub_total (pandas.DataFrame): The total subset DataFrame.
        processed_groups (dict): A dictionary of processed group DataFrames.

    Returns:
        tuple:
            numeric_columns (pandas.Index): The numeric columns in the sub_total DataFrame.
            B (pandas.DataFrame): The 'B' group DataFrame.
    """
    numeric_columns = sub_total.select_dtypes(include=["number"]).columns
    B_union = sub_total[sub_total["grouping"] == "B"]
    B_union = add_missing_level(B_union, "B")

    A_numeric = processed_groups["A"][0].set_index("level")[numeric_columns]
    D_numeric = processed_groups["D"][0].set_index("level")[numeric_columns]

    A_union_D = A_numeric.add(
        D_numeric.reindex(A_numeric.index, fill_value=0), fill_value=0
    )

    B = B_union.set_index("level")[numeric_columns].sub(A_union_D, fill_value=0)

    B_common = pd.DataFrame(
        {"stat": "total", "grouping": "B", "level": ["ha", "ma", "la"]}
    )

    B = B.reset_index().join(B_common, rsuffix="_common")

    return numeric_columns, B


def impute_NDU(B, ha, ma, la, processed_groups, groups):
    """
    Imputes the non-suspended users; values by calculating the difference between the sum of all non-suspended groups and the sum of the 'ha', 'ma', and 'la' groups.

    Args:
        B (pandas.DataFrame): The 'B' group DataFrame.
        ha (pandas.DataFrame): The 'ha' group DataFrame.
        ma (pandas.DataFrame): The 'ma' group DataFrame.
        la (pandas.DataFrame): The 'la' group DataFrame.
        processed_groups (dict): A dictionary of processed group DataFrames.
        groups (list): A list of group names.

    Returns:
        tuple:
            all_levels (pandas.DataFrame): A DataFrame containing all activity levels.
            non_suspended (pandas.DataFrame): A DataFrame containing the sum of all non-suspended groups.
    """
    all_levels = (
        pd.concat([ha, ma, la]).rename(columns={"group": "level"}).set_index("level")
    )
    non_suspended = (
        pd.concat([processed_groups[group][0] for group in groups] + [B])
        .groupby("level")
        .sum(numeric_only=True)
    )
    non_suspended["stat"] = "total"
    return all_levels, non_suspended

def recombine(date, processed_groups, B, suspended):
    """
    Recombines the processed data groups and suspended data into a single DataFrame.

    Args:
        date (datetime): The date for which the data is being processed.
        processed_groups (dict): A dictionary containing the processed data groups.
        B (pd.DataFrame): The B DataFrame.
        suspended (pd.DataFrame): The suspended DataFrame.

    Returns:
        pd.DataFrame: The final DataFrame containing the recombined data.
    """
    dfs_to_concat = []
    for group in processed_groups.values():
        for item in group:
            if isinstance(item, pd.DataFrame):
                dfs_to_concat.append(item)

    if not isinstance(B, pd.DataFrame):
        B = pd.DataFrame(B)

    if date < datetime(2021, 1, 12):
        suspended_p = process_suspended(suspended)
        dfs_to_concat.append(suspended_p)

    exclusive_groups = pd.concat(dfs_to_concat, ignore_index=True)

    return exclusive_groups


def aggregation_func(df_raw, date):
    ha, ma, la, processed_groups, sub_total, groups = preprocessing(df_raw, date)

    numeric_columns, B = pull_B(sub_total, processed_groups)

    all_levels, non_suspended = impute_NDU(B, ha, ma, la, processed_groups, groups)

    suspended = all_levels[numeric_columns].sub(
        non_suspended[numeric_columns], fill_value=0
    )
    suspended["date"] = date

    final = recombine(date, processed_groups, B, suspended)

    return final
```


```python
mut_exclusive_groups = []

for day in mccabe_full["date"].unique():
    mut_exclusive_groups.append(aggregation_func(mccabe_full, day))

total = pd.concat(mut_exclusive_groups).reset_index(drop=True)

# Create a new column 'subsets' by combining 'grouping' and 'level'
total.loc[:, "subsets"] = total["grouping"] + "_" + total["level"]
total = total.drop(columns=["grouping", "level", "stat", "level_common"]).reset_index(
    drop=True
)
# total.to_csv("data/total.csv") checkpoint
```


```python
total = pd.read_csv("data/total.csv").drop(columns=["Unnamed: 0"])
```


```python
total['date'] = pd.to_datetime(total['date'].apply(lambda x: datetime.strptime(x, "%Y-%m-%d").date()))
total['subset_group'] = total['subsets'].apply(lambda x: x.split('_')[0])
total['subset_activity'] = total['subsets'].apply(lambda x: x.split('_')[1])
```

Data collection was an inexhaustive process, so it's important to verify the sudden changes in each groups' behavior. Some changes can be explained by parallel changes in other groups; the number of low activity users sharply declines around July 2020, but at the same time the number of high activity users sharply increases. 


```python
fig, ax = plt.subplots(3, 3, figsize=(15, 10), sharex=True)
for L in total['subset_activity'].unique():
    if L == 'ha':
        i = 0
        title = 'High'
    elif L == 'ma':
        i = 1
        title = 'Moderate'
    else:
        i = 2
        title = 'Low'
    d = total.loc[total['subset_activity'] == L]
    sns.lineplot(d, x='date', y='nusers', hue='subset_group', palette='Set2', ax=ax[0, i])
    sns.lineplot(d, x='date', y='n', hue='subset_group', palette='Set2', ax=ax[1, i])
    sns.lineplot(d, x='date', y='fake_merged', hue='subset_group', palette='Set2', ax=ax[2, i])

fig.text(-0.01, 0.75, 'Number of Users', rotation=90, fontsize=14)
fig.text(-0.01, 0.475, 'Total Tweets', rotation=90, fontsize=14)
fig.text(-0.01, 0.15, 'Fake Tweets', rotation=90, fontsize=14)

fig.text(0.15, 1.01, 'High Activity', fontsize=14)
fig.text(0.475, 1.01, 'Moderate Activity', fontsize=14)
fig.text(0.825, 1.01, 'Low Activity', fontsize=14)

handles, labels = ax[0, 0].get_legend_handles_labels()
for h in handles:
    h.set_linewidth(3)
fig.legend(handles, labels, loc='upper right', ncol=1, bbox_to_anchor=(1.1, 1))

for a in ax.flatten():
    a.xaxis.set_major_locator(mdates.AutoDateLocator())
    a.xaxis.set_major_formatter(mdates.ConciseDateFormatter(a.xaxis.get_major_locator()))
    plt.setp(a.xaxis.get_majorticklabels(), rotation=45, ha='right')
    a.get_legend().remove()
    a.yaxis.set_label_text('')

plt.tight_layout()
plt.show()
```


```python
features = ['fake_merged_initiation', 'fake_merged_rt', 'not_fake_conservative', 'not_fake_liberal', 'not_fake_shopping', 'not_fake_sports', 'n', 'nusers', 'subset_group', 'subset_activity', 'date']

df = total.copy().loc[:, features].pivot_table(index='date', columns=['subset_group', 'subset_activity'], values=['fake_merged_initiation', 'fake_merged_rt', 'not_fake_conservative', 'not_fake_liberal', 'not_fake_shopping', 'not_fake_sports', 'n', 'nusers'], fill_value=0).asfreq('D').fillna(0)
df.columns = ['_'.join(col).strip() for col in df.columns.values]

treatment_start = '2021-01-12'
treatment_end = '2021-01-19'

pretreatment_df = df.loc[:treatment_start]
posttreatment_df = df.loc[treatment_end:]
```


```python
from adtk.detector import MinClusterDetector
from adtk.pipe import Pipeline
from adtk.data import validate_series
from adtk.transformer import PcaProjection
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression

min_cluster_detector = MinClusterDetector(KMeans(n_clusters=4))
steps = [
    ("projection", PcaProjection(k=2)),
    ("detector", min_cluster_detector)
]
pipeline = Pipeline(steps)
pre_treated = validate_series(pretreatment_df)
pre_anomalies = pipeline.fit_detect(pre_treated).reset_index().rename(columns={0: "anomaly"})
```


```python
pre_outlier_data = total.loc[total['date'] < '2021-01-12'].join(pre_anomalies.set_index('date'), on='date')
plt.style.use('seaborn-v0_8-whitegrid')

fig, ax = plt.subplots(1, 3, figsize=(15, 10))

for L in ['ha', 'ma', 'la']:
    if L == 'ha':
        i = 0
        title = 'High'
    elif L == 'ma':
        i = 1
        title = 'Moderate'
    else:
        i = 2
        title = 'Low'
    name = f"{title} Activity Users"
    sns.lineplot(x='date', y='nusers', data=pre_outlier_data.loc[pre_outlier_data['subset_activity'] == L], ax=ax[i], hue='subset_group', palette='Set2')
    sns.scatterplot(x='date', y='nusers', data=pre_outlier_data.loc[(pre_outlier_data['subset_activity'] == L) & (pre_outlier_data['anomaly'] == True)], ax=ax[i], color='red', legend=False, size=800)
    ax[i].set_title(name)
fig.suptitle('Daily # Users', fontsize=16)

handles, labels = ax[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper right', ncol=1, bbox_to_anchor=(1.0, 0.95))
for a in ax:
    a.xaxis.set_major_locator(mdates.MonthLocator())
    a.xaxis.set_major_formatter(mdates.ConciseDateFormatter(a.xaxis.get_major_locator()))
    plt.setp(a.xaxis.get_majorticklabels(), rotation=45, ha='right')
    a.get_legend().remove()
    a.set_ylim(0)
    a.yaxis.set_label_text('')
```

The main outliers seem to be the earliest observations, as well as the general downward trend in November 2020.

## Modeling

To fully understand the network effects of user suspension, I will need to create a synthetic control. 
1) I will first need to create a model of total suspended users, and use time-series forecasting to predict *untreated* total suspended users.
2) I will then use these predictions to predict the number of users and total content for all other groups, using their own lagged values and the suspended users' estimations as predictors.
3) Next, I will model total fake content using the observed pre-treatment data, and use my predictions to estimate the total fake content for the post-treatment period.
4) Finally, I will use the observed and predicted values to calculate the treatment effect.

First I'll explore the structure of the data to determine the best way to model the data.


```python
total = pd.read_csv('total.csv').drop(columns=['Unnamed: 0'])
total[['subset_group', 'subset_activity']] = total['subsets'].str.split('_', expand=True)
total['date'] = pd.to_datetime(total['date'])

features = ['fake_merged_initiation', 'fake_merged_rt', 'not_fake_conservative', 'not_fake_liberal', 'not_fake_shopping', 'not_fake_sports', 'n', 'nusers', 'date', 'subsets', 'subset_group', 'subset_activity']
desm = total.copy()[features].loc[total['date'] > datetime(2019, 12, 8)]

desm['treated'] = desm['date'] >= datetime(2021, 1, 12)
desm['fm'] = desm[['fake_merged_initiation', 'fake_merged_rt']].sum(axis=1)
```


```python
desm['fm_1'] = desm.groupby(['subsets', 'treated'])['fm'].shift(1)
desm['n_1'] = desm.groupby(['subsets', 'treated'])['n'].shift(1)
desm['nusers_1'] = desm.groupby(['subsets', 'treated'])['nusers'].shift(1)
desm_pre = desm.loc[desm['treated'] == False]
```


```python
a = sns.relplot(desm_pre, x='date', y='fm', col='subset_activity', kind='line', hue='subset_group', palette='Set1', height=4, aspect=1.5)
for ax in a.axes.flatten():
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax.xaxis.get_major_locator()))
plt.suptitle('Total Fake Content Over Time', y=1.01)
plt.show()
```


    
![png](cleaned-project-1_files/cleaned-project-1_30_0.png)
    



```python
rolling_desm = desm.loc[desm['treated'] == False].drop(columns='treated').set_index('date').groupby(['subsets', 'subset_group', 'subset_activity']).rolling(window=7).mean().reset_index()
```

I reduced some of the periodicity in the data (either from unobserved components of users' behavior or from the way the data was collected) by using rolling averages with a one-week window.


```python
b = sns.relplot(rolling_desm, x='date', y='fm', col='subset_activity', kind='line', hue='subset_group', palette='Set1', height=4, aspect=1.5, col_order=['ha', 'ma', 'la'])
for ax in b.axes.flatten():
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax.xaxis.get_major_locator()))
plt.suptitle("Smoothed Fake Content Over Time", y=1.02)
plt.show()
```


    
![png](cleaned-project-1_files/cleaned-project-1_33_0.png)
    



```python
sns.relplot(data=rolling_desm, x='fm_1', y='fm', col='subset_activity', kind='scatter', col_wrap=3, col_order=['ha', 'ma', 'la'], height=4, aspect=1, hue='subset_group')
plt.suptitle('Total Fake Content has a Linear-lagged Relationship', y=1.01)
plt.show()
```


    
![png](cleaned-project-1_files/cleaned-project-1_34_0.png)
    



```python
rd_x = rolling_desm.melt(id_vars=['date', 'subset_group', 'subset_activity', 'fm'], value_vars=['not_fake_shopping', 'not_fake_sports', 'not_fake_conservative', 'not_fake_liberal'])
rd_x = rd_x.loc[(rd_x['subset_group'].isin(['D', 'F', 'suspended'])) & (rd_x['subset_activity'].isin(['ha', 'ma']))]
sns.relplot(data=rd_x, x='value', y='fm', hue='variable', col='subset_activity', row='subset_group', kind='scatter')
plt.suptitle('Real Content has a Semi-Linear Relationship with Fake Content', y=1.01)
plt.show()
```


    
![png](cleaned-project-1_files/cleaned-project-1_35_0.png)
    


not_fake_liberal and not_fake_conservative have a linear relationship with fake content in some cases, which indicates group and activity levels are important in modeling


```python
sns.relplot(rolling_desm, x='n', y='n_1', col='subset_activity', hue='subset_group', kind='scatter', col_wrap=3, col_order=['ha', 'ma', 'la'], height=4, aspect=1)
plt.suptitle('Total Content has a Linear-lagged Relationship', y=1.01)
plt.show()
```


    
![png](cleaned-project-1_files/cleaned-project-1_37_0.png)
    


The structural relationships between the different variables are mostly linear and positive.


```python
sns.relplot(data=rolling_desm, x='nusers', y='nusers_1', hue='subset_group', col='subset_activity', facet_kws={'sharey': False}, col_order=['ha', 'ma', 'la'])
plt.suptitle('Total Users has a Mostly Linear-lagged Relationship', y=1.01)
plt.show()
```


    
![png](cleaned-project-1_files/cleaned-project-1_39_0.png)
    


In order to use time-series forecasting, I need to ensure that my data is stationary. Columns that aren't stationary will be differenced to be compatible with my model.


```python
from statsmodels.tsa.stattools import adfuller
```


```python
X = ['not_fake_conservative', 'not_fake_liberal', 'not_fake_shopping', 'not_fake_sports']
Y = ['fm']
Z = ['n', 'nusers']
all_vars = X + Y + Z
pdf = desm_pre.copy()[all_vars + ['subset_group', 'subset_activity', 'date']].loc[desm_pre['date'] > datetime(2019, 12, 18)]
pdf.reset_index(inplace=True, drop=True)
pdf.set_index(['date', 'subset_group', 'subset_activity'], inplace=True)
```


```python
def adf_test(col):
    result = adfuller(col.dropna())
    return result[1] < 0.05
for var in all_vars:
    if adf_test(pdf[var]):
        print(f'{var} is stationary')
    else:
        print(f'{var} is not stationary')
```

    not_fake_conservative is stationary
    not_fake_liberal is stationary
    not_fake_shopping is stationary
    not_fake_sports is stationary
    fm is stationary
    n is stationary
    nusers is stationary


All of the variables are stationary.

### Modeling Suspended Users


```python
from statsmodels.tsa.api import VAR

suspended = pdf.loc[(pdf.index.get_level_values(1) == 'suspended') & (pdf.index.get_level_values(0) < datetime(2021, 1, 6))]
```


```python
s_nusers_wide = suspended.reset_index().pivot(index='date', columns='subset_activity', values='nusers')
s_nusers_wide.index = pd.to_datetime(s_nusers_wide.index)
for col in s_nusers_wide.columns:
    s_nusers_wide[col] = np.sqrt(s_nusers_wide[col])
model = VAR(s_nusers_wide, freq='D')
lag_selection = model.select_order(maxlags=10)
lag_selection.summary()
```




<table class="simpletable">
<caption>VAR Order Selection (* highlights the minimums)</caption>
<tr>
   <td></td>      <th>AIC</th>         <th>BIC</th>         <th>FPE</th>        <th>HQIC</th>    
</tr>
<tr>
  <th>0</th>  <td>     2.896</td>  <td>     2.928</td>  <td>     18.11</td>  <td>     2.909</td> 
</tr>
<tr>
  <th>1</th>  <td>    -1.724</td>  <td>    -1.598</td>  <td>    0.1784</td>  <td>    -1.674</td> 
</tr>
<tr>
  <th>2</th>  <td>    -1.922</td>  <td>    -1.701</td>  <td>    0.1464</td>  <td>    -1.834</td> 
</tr>
<tr>
  <th>3</th>  <td>    -2.026</td>  <td>    -1.711*</td> <td>    0.1319</td>  <td>    -1.901*</td>
</tr>
<tr>
  <th>4</th>  <td>    -2.025</td>  <td>    -1.616</td>  <td>    0.1320</td>  <td>    -1.862</td> 
</tr>
<tr>
  <th>5</th>  <td>    -2.042</td>  <td>    -1.539</td>  <td>    0.1298</td>  <td>    -1.842</td> 
</tr>
<tr>
  <th>6</th>  <td>    -2.077</td>  <td>    -1.479</td>  <td>    0.1253</td>  <td>    -1.840</td> 
</tr>
<tr>
  <th>7</th>  <td>    -2.110*</td> <td>    -1.418</td>  <td>    0.1213*</td> <td>    -1.835</td> 
</tr>
<tr>
  <th>8</th>  <td>    -2.088</td>  <td>    -1.301</td>  <td>    0.1240</td>  <td>    -1.775</td> 
</tr>
<tr>
  <th>9</th>  <td>    -2.054</td>  <td>    -1.173</td>  <td>    0.1283</td>  <td>    -1.705</td> 
</tr>
<tr>
  <th>10</th> <td>    -2.054</td>  <td>    -1.078</td>  <td>    0.1284</td>  <td>    -1.667</td> 
</tr>
</table>



A 3 day lag seems best. We start the prediction range earlier than needed to allow the predictions to adjust to the lag appropriately.


```python
result = model.fit(7)
test_predictions = pd.DataFrame(result.forecast(s_nusers_wide.values[-50:], steps=60), columns=['pred_ha', 'pred_la', 'pred_ma'])
preds_real = s_nusers_wide.iloc[-50:]
test_predictions.index = pd.date_range('2020-11-11', periods=60)
combined = test_predictions.join(preds_real)

fig, ax = plt.subplots(3, 1, figsize=(15, 12))
for i, col in enumerate(['ha', 'ma', 'la']):
    sns.lineplot(data=combined, x=combined.index, y=f'pred_{col}', ax=ax[i], label='Predicted')
    sns.lineplot(data=combined, x=combined.index, y=col, ax=ax[i], label='Observed')
    ax[i].set_title(f'{col.upper()[0]} Activity Users')
    ax[i].set_ylabel('Number of Users')
    if i != 2:
        ax[i].set_xticklabels([])
        ax[i].set_xlabel('')
ax[2].set_xlabel('Date')
plt.show()
```


    
![png](cleaned-project-1_files/cleaned-project-1_49_0.png)
    


The fit seems appropriate for the data. Now I can construct the first part of the synthetic control.


```python
synthetic = pd.DataFrame(result.forecast(test_predictions.values[-7:], steps=157))
synthetic.columns = ['sus_ha_nusers', 'sus_ma_nusers', 'sus_la_nusers']
```


```python
suspended_data = pdf.loc[pdf.index.get_level_values(1) == 'suspended', ['nusers', 'n', 'fm', 'not_fake_conservative', 'not_fake_liberal', 'not_fake_shopping', 'not_fake_sports']].reset_index()
```

Using the predicted numbers of users, I will now predict values for other features. These predictions will provide more depth for the final model of all users' behavior. Since 'not fake sports' has a relatively low correlation with the target variable, I will not use it in the final model, but all other features will be used.


```python
sns.lmplot(x='nusers', y='n', hue='subset_activity', data=suspended_data)
plt.title('Number of Suspended Users and Sum of Activity by Activity Levels')
plt.show()
```


    
![png](cleaned-project-1_files/cleaned-project-1_54_0.png)
    


It seems like low- and moderate-activity users have very similar slopes, but high-activity users have a steeper, curved slope.


```python
from sklearn.linear_model import BayesianRidge
from sklearn.model_selection import train_test_split, GridSearchCV

suspended_subset = suspended_data.loc[(suspended_data['date'] >= datetime(2020, 1, 1)) & (suspended_data['date'] <= datetime(2021, 1, 1))] # select one-year period close to (not including) onset of treatment
suspended_subset['high_activity'] = (suspended_subset['subset_activity'] == 'ha').astype(int) # create binary variable for high activity
X = suspended_subset[['high_activity', 'nusers']]
Y = suspended_subset['n']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=25)

model = BayesianRidge(lambda_2=1e-3, alpha_1=1e-3, tol=1e-4, alpha_2=5e-5)
params = {
    'lambda_1': np.linspace(1e-10, 6e-10, 7),
}
grid_search = GridSearchCV(model, params, cv=4)
grid_search.fit(X_train, Y_train)
best_model = grid_search.best_estimator_
```


```python
print(f"Training score: {best_model.score(X_train, Y_train)}, Test score: {best_model.score(X_test, Y_test)}")
```

    Training score: 0.8763366599345839, Test score: 0.8980982570396228


Now I can use this model to predict total content for suspended users using my predictions for number of users.


```python
def suspended_n_prediction(activity):
    nusers = synthetic[[f'sus_{activity}_nusers']]
    nusers.columns = ['nusers']
    if activity == 'ha':
        high_activity = 1
    else:
        high_activity = 0
    nusers['high_activity'] = high_activity
    nusers = nusers.reindex(columns=['high_activity', 'nusers'])
    return best_model.predict(nusers)

synthetic['suspended_ha_n'] = suspended_n_prediction('ha')
synthetic['suspended_ma_n'] = suspended_n_prediction('ma')
synthetic['suspended_la_n'] = suspended_n_prediction('la')
```


```python
suspended_subset.reset_index().set_index(['subset_activity', 'date']).shift(1).reset_index()
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
      <th>subset_activity</th>
      <th>date</th>
      <th>index</th>
      <th>subset_group</th>
      <th>nusers</th>
      <th>n</th>
      <th>fm</th>
      <th>not_fake_conservative</th>
      <th>not_fake_liberal</th>
      <th>not_fake_shopping</th>
      <th>not_fake_sports</th>
      <th>high_activity</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>ha</td>
      <td>2020-01-01</td>
      <td>NaN</td>
      <td>None</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>la</td>
      <td>2020-01-01</td>
      <td>39.0</td>
      <td>suspended</td>
      <td>170.0</td>
      <td>2353.0</td>
      <td>726.0</td>
      <td>194.0</td>
      <td>37.0</td>
      <td>5.0</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>ma</td>
      <td>2020-01-01</td>
      <td>40.0</td>
      <td>suspended</td>
      <td>7.0</td>
      <td>11.0</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>ha</td>
      <td>2020-01-02</td>
      <td>41.0</td>
      <td>suspended</td>
      <td>132.0</td>
      <td>524.0</td>
      <td>112.0</td>
      <td>43.0</td>
      <td>13.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>la</td>
      <td>2020-01-02</td>
      <td>42.0</td>
      <td>suspended</td>
      <td>175.0</td>
      <td>3164.0</td>
      <td>923.0</td>
      <td>356.0</td>
      <td>47.0</td>
      <td>4.0</td>
      <td>3.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>1096</th>
      <td>la</td>
      <td>2020-12-31</td>
      <td>1134.0</td>
      <td>suspended</td>
      <td>178.0</td>
      <td>4672.0</td>
      <td>1632.0</td>
      <td>405.0</td>
      <td>22.0</td>
      <td>4.0</td>
      <td>6.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>1097</th>
      <td>ma</td>
      <td>2020-12-31</td>
      <td>1135.0</td>
      <td>suspended</td>
      <td>75.0</td>
      <td>732.0</td>
      <td>252.0</td>
      <td>55.0</td>
      <td>6.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1098</th>
      <td>ha</td>
      <td>2021-01-01</td>
      <td>1136.0</td>
      <td>suspended</td>
      <td>177.0</td>
      <td>1778.0</td>
      <td>593.0</td>
      <td>149.0</td>
      <td>18.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1099</th>
      <td>la</td>
      <td>2021-01-01</td>
      <td>1137.0</td>
      <td>suspended</td>
      <td>181.0</td>
      <td>5006.0</td>
      <td>1547.0</td>
      <td>323.0</td>
      <td>13.0</td>
      <td>9.0</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>1100</th>
      <td>ma</td>
      <td>2021-01-01</td>
      <td>1138.0</td>
      <td>suspended</td>
      <td>66.0</td>
      <td>907.0</td>
      <td>262.0</td>
      <td>36.0</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
<p>1101 rows × 12 columns</p>
</div>




```python
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor

X = suspended_subset[['subset_activity', 'nusers', 'n']]
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
      <th>date</th>
      <th>subset_group</th>
      <th>subset_activity</th>
      <th>nusers</th>
      <th>n</th>
      <th>fm</th>
      <th>not_fake_conservative</th>
      <th>not_fake_liberal</th>
      <th>not_fake_shopping</th>
      <th>not_fake_sports</th>
      <th>high_activity</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>39</th>
      <td>2020-01-01</td>
      <td>suspended</td>
      <td>ha</td>
      <td>170</td>
      <td>2353.0</td>
      <td>726.0</td>
      <td>194.0</td>
      <td>37.0</td>
      <td>5.0</td>
      <td>0.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>40</th>
      <td>2020-01-01</td>
      <td>suspended</td>
      <td>la</td>
      <td>7</td>
      <td>11.0</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>41</th>
      <td>2020-01-01</td>
      <td>suspended</td>
      <td>ma</td>
      <td>132</td>
      <td>524.0</td>
      <td>112.0</td>
      <td>43.0</td>
      <td>13.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>42</th>
      <td>2020-01-02</td>
      <td>suspended</td>
      <td>ha</td>
      <td>175</td>
      <td>3164.0</td>
      <td>923.0</td>
      <td>356.0</td>
      <td>47.0</td>
      <td>4.0</td>
      <td>3.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>43</th>
      <td>2020-01-02</td>
      <td>suspended</td>
      <td>la</td>
      <td>9</td>
      <td>14.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>1135</th>
      <td>2020-12-31</td>
      <td>suspended</td>
      <td>la</td>
      <td>75</td>
      <td>732.0</td>
      <td>252.0</td>
      <td>55.0</td>
      <td>6.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1136</th>
      <td>2020-12-31</td>
      <td>suspended</td>
      <td>ma</td>
      <td>177</td>
      <td>1778.0</td>
      <td>593.0</td>
      <td>149.0</td>
      <td>18.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1137</th>
      <td>2021-01-01</td>
      <td>suspended</td>
      <td>ha</td>
      <td>181</td>
      <td>5006.0</td>
      <td>1547.0</td>
      <td>323.0</td>
      <td>13.0</td>
      <td>9.0</td>
      <td>0.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1138</th>
      <td>2021-01-01</td>
      <td>suspended</td>
      <td>la</td>
      <td>66</td>
      <td>907.0</td>
      <td>262.0</td>
      <td>36.0</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1139</th>
      <td>2021-01-01</td>
      <td>suspended</td>
      <td>ma</td>
      <td>184</td>
      <td>1600.0</td>
      <td>467.0</td>
      <td>72.0</td>
      <td>10.0</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>1101 rows × 11 columns</p>
</div>



### Modeling Other Users


```python
wide_table_all = pd.DataFrame(pdf.stack().reset_index()).pivot_table(
    index='date',
    columns=['subset_group', 'subset_activity', 'level_3'],
    values=0
)
wide_table_all.columns = ['_'.join(col).strip() for col in wide_table_all.columns.values]
```


```python
wide_lag = wide_table_all.shift(1)
wide_lag.columns = [f'{col}_lag_1' for col in wide_lag.columns]
wide_lag_2 = wide_table_all.shift(2)
wide_lag_2.columns = [f'{col}_lag_2' for col in wide_lag_2.columns]
wide_subset = wide_table_all.loc[(wide_table_all.index > datetime(2020, 1, 1)) & (wide_table_all.index < datetime(2021, 1, 1))].join(wide_lag, how='inner').join(wide_lag_2, how='inner')
```

The basic relationships of the variables can be described as:



```python
def mm(graph):
    graphbytes = graph.encode("utf8")
    base64_bytes = base64.urlsafe_b64encode(graphbytes)
    base64_string = base64_bytes.decode("ascii")
    img = im.open(io.BytesIO(requests.get('https://mermaid.ink/img/' + base64_string).content))
    plt.imshow(img)
    plt.axis('off')

mm("""
    graph TD
    a[nusers] --> b[n]
    c[fm]
    d[not_fake_conservative]
    e[not_fake_liberal]
    f[not_fake_shopping]
    g[not_fake_sports]
    b --> d & e & f & g
    b ----> c
    d & e & f & g --> c
    """)
```


    
![png](cleaned-project-1_files/cleaned-project-1_66_0.png)
    


I will standardize number of users and total content, but the other features are all on different scales. To allow for better comparisons without introducing bias, I will standardize the other features relative to the total content put out by each group.


```python
def standardize_group(subset, activity_level, df_):
    result = {}
    df = df_.copy().reset_index(drop=True)
    for suffix in ['', '_lag_1', '_lag_2']:
        n_col = f"{subset}_{activity_level}_n{suffix}"
        nusers_col = f"{subset}_{activity_level}_nusers{suffix}"
        if n_col in df.columns and nusers_col in df.columns:
            n = df[n_col]
            nusers = df[nusers_col]
            n_per_user = n / nusers.replace(0, np.nan)
            n_per_user = n_per_user.fillna(0)

            result[n_col] = n_per_user
            result[nusers_col] = nusers

            prefix = f"{subset}_{activity_level}_"
            for col in df.columns:
                if col.startswith(prefix) and col.endswith(suffix) and col not in [n_col, nusers_col]:
                    new_col_name = col
                    _values = df[col]
                    standardized_values = _values / n_per_user.replace(0, np.nan)
                    standardized_values = standardized_values.fillna(0)
                    result[new_col_name] = standardized_values

    return pd.DataFrame(result)
```


```python
standardized_columns = []
for subset in ['A', 'B', 'D', 'F', 'suspended', 'nfns']:
    for activity_level in ['ha', 'ma', 'la']:
        sub = standardize_group(subset, activity_level, wide_subset)
        standardized_columns.append(sub)
standardized_df = pd.concat(standardized_columns, axis=1)
```


```python
nusers_columns = [c for c in standardized_df.columns if 'nusers' in c]
nusers_lag_columns = [c for c in nusers_columns if 'lag' in c]
suspended_columns = [c for c in standardized_df.columns if ('suspended' in c) & (re.search(r'_n(?=[_\b])', c) is not None)]
X_ = standardized_df[list(set(nusers_lag_columns + suspended_columns))]
Y = standardized_df[[c for c in nusers_columns if (c not in nusers_lag_columns) and (c not in suspended_columns)]]
```


```python
from sklearn.svm import NuSVR
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X = scaler.fit_transform(X_)
y = scaler.fit_transform(Y)
X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2, random_state=40)
model = NuSVR()
params = {
    'C': [0.25, 0.5, 1],
    'nu': [0.1, 0.25, 0.5],
    'kernel': ['linear', 'poly', 'rbf'],
    'tol': [1e-5, 1e-4, 1e-3],
    'degree': [2, 3, 4]
}
grid_search = GridSearchCV(model, params, cv=3, verbose=1, n_jobs=3, pre_dispatch='2*n_jobs')
mr = MultiOutputRegressor(grid_search)
mr.fit(X_train, Y_train)
```

    Fitting 3 folds for each of 243 candidates, totalling 729 fits
    Fitting 3 folds for each of 243 candidates, totalling 729 fits
    Fitting 3 folds for each of 243 candidates, totalling 729 fits
    Fitting 3 folds for each of 243 candidates, totalling 729 fits
    Fitting 3 folds for each of 243 candidates, totalling 729 fits
    Fitting 3 folds for each of 243 candidates, totalling 729 fits
    Fitting 3 folds for each of 243 candidates, totalling 729 fits
    Fitting 3 folds for each of 243 candidates, totalling 729 fits
    Fitting 3 folds for each of 243 candidates, totalling 729 fits
    Fitting 3 folds for each of 243 candidates, totalling 729 fits
    Fitting 3 folds for each of 243 candidates, totalling 729 fits
    Fitting 3 folds for each of 243 candidates, totalling 729 fits
    Fitting 3 folds for each of 243 candidates, totalling 729 fits
    Fitting 3 folds for each of 243 candidates, totalling 729 fits
    Fitting 3 folds for each of 243 candidates, totalling 729 fits





<style>#sk-container-id-37 {
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

#sk-container-id-37 {
  color: var(--sklearn-color-text);
}

#sk-container-id-37 pre {
  padding: 0;
}

#sk-container-id-37 input.sk-hidden--visually {
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

#sk-container-id-37 div.sk-dashed-wrapped {
  border: 1px dashed var(--sklearn-color-line);
  margin: 0 0.4em 0.5em 0.4em;
  box-sizing: border-box;
  padding-bottom: 0.4em;
  background-color: var(--sklearn-color-background);
}

#sk-container-id-37 div.sk-container {
  /* jupyter's `normalize.less` sets `[hidden] { display: none; }`
     but bootstrap.min.css set `[hidden] { display: none !important; }`
     so we also need the `!important` here to be able to override the
     default hidden behavior on the sphinx rendered scikit-learn.org.
     See: https://github.com/scikit-learn/scikit-learn/issues/21755 */
  display: inline-block !important;
  position: relative;
}

#sk-container-id-37 div.sk-text-repr-fallback {
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

#sk-container-id-37 div.sk-parallel-item::after {
  content: "";
  width: 100%;
  border-bottom: 2px solid var(--sklearn-color-text-on-default-background);
  flex-grow: 1;
}

#sk-container-id-37 div.sk-parallel {
  display: flex;
  align-items: stretch;
  justify-content: center;
  background-color: var(--sklearn-color-background);
  position: relative;
}

#sk-container-id-37 div.sk-parallel-item {
  display: flex;
  flex-direction: column;
}

#sk-container-id-37 div.sk-parallel-item:first-child::after {
  align-self: flex-end;
  width: 50%;
}

#sk-container-id-37 div.sk-parallel-item:last-child::after {
  align-self: flex-start;
  width: 50%;
}

#sk-container-id-37 div.sk-parallel-item:only-child::after {
  width: 0;
}

/* Serial-specific style estimator block */

#sk-container-id-37 div.sk-serial {
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

#sk-container-id-37 div.sk-toggleable {
  /* Default theme specific background. It is overwritten whether we have a
  specific estimator or a Pipeline/ColumnTransformer */
  background-color: var(--sklearn-color-background);
}

/* Toggleable label */
#sk-container-id-37 label.sk-toggleable__label {
  cursor: pointer;
  display: block;
  width: 100%;
  margin-bottom: 0;
  padding: 0.5em;
  box-sizing: border-box;
  text-align: center;
}

#sk-container-id-37 label.sk-toggleable__label-arrow:before {
  /* Arrow on the left of the label */
  content: "▸";
  float: left;
  margin-right: 0.25em;
  color: var(--sklearn-color-icon);
}

#sk-container-id-37 label.sk-toggleable__label-arrow:hover:before {
  color: var(--sklearn-color-text);
}

/* Toggleable content - dropdown */

#sk-container-id-37 div.sk-toggleable__content {
  max-height: 0;
  max-width: 0;
  overflow: hidden;
  text-align: left;
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-0);
}

#sk-container-id-37 div.sk-toggleable__content.fitted {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-0);
}

#sk-container-id-37 div.sk-toggleable__content pre {
  margin: 0.2em;
  border-radius: 0.25em;
  color: var(--sklearn-color-text);
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-0);
}

#sk-container-id-37 div.sk-toggleable__content.fitted pre {
  /* unfitted */
  background-color: var(--sklearn-color-fitted-level-0);
}

#sk-container-id-37 input.sk-toggleable__control:checked~div.sk-toggleable__content {
  /* Expand drop-down */
  max-height: 200px;
  max-width: 100%;
  overflow: auto;
}

#sk-container-id-37 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {
  content: "▾";
}

/* Pipeline/ColumnTransformer-specific style */

#sk-container-id-37 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {
  color: var(--sklearn-color-text);
  background-color: var(--sklearn-color-unfitted-level-2);
}

#sk-container-id-37 div.sk-label.fitted input.sk-toggleable__control:checked~label.sk-toggleable__label {
  background-color: var(--sklearn-color-fitted-level-2);
}

/* Estimator-specific style */

/* Colorize estimator box */
#sk-container-id-37 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-2);
}

#sk-container-id-37 div.sk-estimator.fitted input.sk-toggleable__control:checked~label.sk-toggleable__label {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-2);
}

#sk-container-id-37 div.sk-label label.sk-toggleable__label,
#sk-container-id-37 div.sk-label label {
  /* The background is the default theme color */
  color: var(--sklearn-color-text-on-default-background);
}

/* On hover, darken the color of the background */
#sk-container-id-37 div.sk-label:hover label.sk-toggleable__label {
  color: var(--sklearn-color-text);
  background-color: var(--sklearn-color-unfitted-level-2);
}

/* Label box, darken color on hover, fitted */
#sk-container-id-37 div.sk-label.fitted:hover label.sk-toggleable__label.fitted {
  color: var(--sklearn-color-text);
  background-color: var(--sklearn-color-fitted-level-2);
}

/* Estimator label */

#sk-container-id-37 div.sk-label label {
  font-family: monospace;
  font-weight: bold;
  display: inline-block;
  line-height: 1.2em;
}

#sk-container-id-37 div.sk-label-container {
  text-align: center;
}

/* Estimator-specific */
#sk-container-id-37 div.sk-estimator {
  font-family: monospace;
  border: 1px dotted var(--sklearn-color-border-box);
  border-radius: 0.25em;
  box-sizing: border-box;
  margin-bottom: 0.5em;
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-0);
}

#sk-container-id-37 div.sk-estimator.fitted {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-0);
}

/* on hover */
#sk-container-id-37 div.sk-estimator:hover {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-2);
}

#sk-container-id-37 div.sk-estimator.fitted:hover {
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

#sk-container-id-37 a.estimator_doc_link {
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

#sk-container-id-37 a.estimator_doc_link.fitted {
  /* fitted */
  border: var(--sklearn-color-fitted-level-1) 1pt solid;
  color: var(--sklearn-color-fitted-level-1);
}

/* On hover */
#sk-container-id-37 a.estimator_doc_link:hover {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-3);
  color: var(--sklearn-color-background);
  text-decoration: none;
}

#sk-container-id-37 a.estimator_doc_link.fitted:hover {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-3);
}
</style><div id="sk-container-id-37" class="sk-top-container"><div class="sk-text-repr-fallback"><pre>MultiOutputRegressor(estimator=GridSearchCV(cv=3, estimator=NuSVR(), n_jobs=3,
                                            param_grid={&#x27;C&#x27;: [0.25, 0.5, 1],
                                                        &#x27;degree&#x27;: [2, 3, 4],
                                                        &#x27;kernel&#x27;: [&#x27;linear&#x27;,
                                                                   &#x27;poly&#x27;,
                                                                   &#x27;rbf&#x27;],
                                                        &#x27;nu&#x27;: [0.1, 0.25, 0.5],
                                                        &#x27;tol&#x27;: [1e-05, 0.0001,
                                                                0.001]},
                                            verbose=1))</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class="sk-container" hidden><div class="sk-item sk-dashed-wrapped"><div class="sk-label-container"><div class="sk-label fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-127" type="checkbox" ><label for="sk-estimator-id-127" class="sk-toggleable__label fitted sk-toggleable__label-arrow fitted">&nbsp;&nbsp;MultiOutputRegressor<a class="sk-estimator-doc-link fitted" rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.5/modules/generated/sklearn.multioutput.MultiOutputRegressor.html">?<span>Documentation for MultiOutputRegressor</span></a><span class="sk-estimator-doc-link fitted">i<span>Fitted</span></span></label><div class="sk-toggleable__content fitted"><pre>MultiOutputRegressor(estimator=GridSearchCV(cv=3, estimator=NuSVR(), n_jobs=3,
                                            param_grid={&#x27;C&#x27;: [0.25, 0.5, 1],
                                                        &#x27;degree&#x27;: [2, 3, 4],
                                                        &#x27;kernel&#x27;: [&#x27;linear&#x27;,
                                                                   &#x27;poly&#x27;,
                                                                   &#x27;rbf&#x27;],
                                                        &#x27;nu&#x27;: [0.1, 0.25, 0.5],
                                                        &#x27;tol&#x27;: [1e-05, 0.0001,
                                                                0.001]},
                                            verbose=1))</pre></div> </div></div><div class="sk-parallel"><div class="sk-parallel-item"><div class="sk-item"><div class="sk-label-container"><div class="sk-label fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-128" type="checkbox" ><label for="sk-estimator-id-128" class="sk-toggleable__label fitted sk-toggleable__label-arrow fitted">estimator: GridSearchCV</label><div class="sk-toggleable__content fitted"><pre>GridSearchCV(cv=3, estimator=NuSVR(), n_jobs=3,
             param_grid={&#x27;C&#x27;: [0.25, 0.5, 1], &#x27;degree&#x27;: [2, 3, 4],
                         &#x27;kernel&#x27;: [&#x27;linear&#x27;, &#x27;poly&#x27;, &#x27;rbf&#x27;],
                         &#x27;nu&#x27;: [0.1, 0.25, 0.5],
                         &#x27;tol&#x27;: [1e-05, 0.0001, 0.001]},
             verbose=1)</pre></div> </div></div><div class="sk-serial"><div class="sk-item sk-dashed-wrapped"><div class="sk-parallel"><div class="sk-parallel-item"><div class="sk-item"><div class="sk-label-container"><div class="sk-label fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-129" type="checkbox" ><label for="sk-estimator-id-129" class="sk-toggleable__label fitted sk-toggleable__label-arrow fitted">estimator: NuSVR</label><div class="sk-toggleable__content fitted"><pre>NuSVR()</pre></div> </div></div><div class="sk-serial"><div class="sk-item"><div class="sk-estimator fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-130" type="checkbox" ><label for="sk-estimator-id-130" class="sk-toggleable__label fitted sk-toggleable__label-arrow fitted">&nbsp;NuSVR<a class="sk-estimator-doc-link fitted" rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.5/modules/generated/sklearn.svm.NuSVR.html">?<span>Documentation for NuSVR</span></a></label><div class="sk-toggleable__content fitted"><pre>NuSVR()</pre></div> </div></div></div></div></div></div></div></div></div></div></div></div></div></div>




```python
mr.score(X_train, Y_train), mr.score(X_test, Y_test)
```




    (0.7528409927367807, 0.5948398836799705)




```python
[mr.estimators_[i].best_estimator_.epsilon for i in range(len(mr.estimators_))]
```




    [10.0,
     0.00046415888336127535,
     0.00046415888336127535,
     10.0,
     0.00046415888336127535,
     0.00046415888336127535,
     2.154434690031878e-08,
     0.00046415888336127535,
     0.00046415888336127535,
     2.154434690031878e-08,
     1e-25,
     2.154434690031878e-08,
     1e-12,
     0.00046415888336127535,
     1e-25]




```python
synth = synthetic.copy()
synth['suspended_ha_n'] = synth['sus_ha_nusers'] / synth['sus_ha_n']
synth['suspended_ma_n'] = synth['sus_ma_nusers'] / synth['sus_ma_n']
synth['suspended_la_n'] = synth['sus_la_nusers'] / synth['sus_la_n']
synth.columns = [c.replace('sus_', 'suspended_') if c.endswith('_nusers') else c for c in synth.columns]
synth_lag1 = synth.shift(1)
synth_lag2 = synth.shift(2)
synth_lag1.columns = [f"{col}_lag_1" for col in synth_lag1.columns]
synth_lag2.columns = [f"{col}_lag_2" for col in synth_lag2.columns]
synthet = synth.join(synth_lag1).join(synth_lag2).iloc[7:].reset_index(drop=True)[[c for c in synth.columns if not c.startswith('sus_')]]
scaler = StandardScaler()
synthet_standard = scaler.fit_transform(synthet)
synthe = pd.DataFrame(synthet_standard, columns=synthet.columns)
```


```python
series = standardized_df[[c for c in X_.columns]].iloc[-1:].reset_index(drop=True)
full_synth = pd.concat([synthe, series[[c for c in series.columns if c not in suspended_columns]]], axis=1)
```


```python
lag_2 = [c for c in series.columns if 'lag_2' in c]
lag_1 = [c for c in series.columns if 'lag_1' in c]

def prepare_row(last_row, row):
    lag_1_2 = last_row[lag_1]
    lag_1_ = last_row[[c for c in series.columns if (c not in lag_1) & (c not in lag_2)]]
    lag_1_2.index = [c.replace('lag_1', 'lag_2') for c in lag_1_2.index]
    lag_1_.index = [f"{c}_lag_1" for c in lag_1_.index]
    row_sus = row[suspended_columns]
    next_row = pd.concat([lag_1_, lag_1_2, row_sus], axis=1)
    predictions = mr.predict(next_row)
    y_cols = [c for c in nusers_columns if (c not in nusers_lag_columns) and (c not in suspended_columns)]
    predictions = pd.DataFrame(predictions, columns=y_cols)
    full_row = pd.concat([row, predictions], axis=1)
    return full_row
prepare_row(full_synth.iloc[-1], full_synth.iloc[0])
```


    ---------------------------------------------------------------------------

    KeyError                                  Traceback (most recent call last)

    Cell In[291], line 16
         14     full_row = pd.concat([row, predictions], axis=1)
         15     return full_row
    ---> 16 prepare_row(full_synth.iloc[-1], full_synth.iloc[0])


    Cell In[291], line 5, in prepare_row(last_row, row)
          4 def prepare_row(last_row, row):
    ----> 5     lag_1_2 = last_row[lag_1]
          6     lag_1_ = last_row[[c for c in series.columns if (c not in lag_1) & (c not in lag_2)]]
          7     lag_1_2.index = [c.replace('lag_1', 'lag_2') for c in lag_1_2.index]


    File ~/.pyenv/versions/twitter/lib/python3.10/site-packages/pandas/core/series.py:1153, in Series.__getitem__(self, key)
       1150     key = np.asarray(key, dtype=bool)
       1151     return self._get_rows_with_mask(key)
    -> 1153 return self._get_with(key)


    File ~/.pyenv/versions/twitter/lib/python3.10/site-packages/pandas/core/series.py:1194, in Series._get_with(self, key)
       1191         return self.iloc[key]
       1193 # handle the dup indexing case GH#4246
    -> 1194 return self.loc[key]


    File ~/.pyenv/versions/twitter/lib/python3.10/site-packages/pandas/core/indexing.py:1191, in _LocationIndexer.__getitem__(self, key)
       1189 maybe_callable = com.apply_if_callable(key, self.obj)
       1190 maybe_callable = self._check_deprecated_callable_usage(key, maybe_callable)
    -> 1191 return self._getitem_axis(maybe_callable, axis=axis)


    File ~/.pyenv/versions/twitter/lib/python3.10/site-packages/pandas/core/indexing.py:1420, in _LocIndexer._getitem_axis(self, key, axis)
       1417     if hasattr(key, "ndim") and key.ndim > 1:
       1418         raise ValueError("Cannot index with multidimensional key")
    -> 1420     return self._getitem_iterable(key, axis=axis)
       1422 # nested tuple slicing
       1423 if is_nested_tuple(key, labels):


    File ~/.pyenv/versions/twitter/lib/python3.10/site-packages/pandas/core/indexing.py:1360, in _LocIndexer._getitem_iterable(self, key, axis)
       1357 self._validate_key(key, axis)
       1359 # A collection of keys
    -> 1360 keyarr, indexer = self._get_listlike_indexer(key, axis)
       1361 return self.obj._reindex_with_indexers(
       1362     {axis: [keyarr, indexer]}, copy=True, allow_dups=True
       1363 )


    File ~/.pyenv/versions/twitter/lib/python3.10/site-packages/pandas/core/indexing.py:1558, in _LocIndexer._get_listlike_indexer(self, key, axis)
       1555 ax = self.obj._get_axis(axis)
       1556 axis_name = self.obj._get_axis_name(axis)
    -> 1558 keyarr, indexer = ax._get_indexer_strict(key, axis_name)
       1560 return keyarr, indexer


    File ~/.pyenv/versions/twitter/lib/python3.10/site-packages/pandas/core/indexes/base.py:6200, in Index._get_indexer_strict(self, key, axis_name)
       6197 else:
       6198     keyarr, indexer, new_indexer = self._reindex_non_unique(keyarr)
    -> 6200 self._raise_if_missing(keyarr, indexer, axis_name)
       6202 keyarr = self.take(indexer)
       6203 if isinstance(key, Index):
       6204     # GH 42790 - Preserve name from an Index


    File ~/.pyenv/versions/twitter/lib/python3.10/site-packages/pandas/core/indexes/base.py:6252, in Index._raise_if_missing(self, key, indexer, axis_name)
       6249     raise KeyError(f"None of [{key}] are in the [{axis_name}]")
       6251 not_found = list(ensure_index(key)[missing_mask.nonzero()[0]].unique())
    -> 6252 raise KeyError(f"{not_found} not in index")


    KeyError: "['suspended_la_not_fake_sports_lag_1', 'suspended_ha_not_fake_sports_lag_1', 'suspended_ha_not_fake_conservative_lag_1', 'suspended_ma_nusers_lag_1', 'suspended_ma_not_fake_conservative_lag_1', 'suspended_ha_nusers_lag_1', 'suspended_la_nusers_lag_1', 'suspended_la_not_fake_conservative_lag_1', 'suspended_ha_not_fake_liberal_lag_1', 'suspended_la_not_fake_liberal_lag_1', 'suspended_la_n_lag_1', 'suspended_ha_n_lag_1', 'suspended_ma_n_lag_1', 'suspended_ma_not_fake_shopping_lag_1', 'suspended_ha_not_fake_shopping_lag_1', 'suspended_ma_not_fake_sports_lag_1', 'suspended_ma_not_fake_liberal_lag_1', 'suspended_la_not_fake_shopping_lag_1'] not in index"


## Sources
McCabe, S.D., Ferrari, D., Green, J. et al. Post-January 6th deplatforming reduced the reach of misinformation on Twitter. Nature 630, 132–140 (2024). https://doi.org/10.1038/s41586-024-07524-8


