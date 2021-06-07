#%%
import pandas as pd
import numpy as np
from plotnine import *
import warnings

warnings.filterwarnings("ignore")

# constant
bonus_bar = 1000

# data wrangling
df = (
    pd.read_csv("./data/training.csv")
    .drop("junk", axis=1)
    .rename(columns={"fav food": "fav_food", "income ": "income"})
    .assign(bonus=lambda x: x.income / 100, total=lambda x: x.income + x.bonus)
    .query("income.notnull() and bonus >= @bonus_bar")
    .loc[:, ["education", "income"]]
    .groupby(["education"])
    .income.agg(["count", "min", "max", "mean"])
    .round(2)
    .sort_values("mean", ascending=False)
    .assign(note=lambda x: np.where(x["mean"] > 230000, 1, 0))
)

# data visualization
plot = (
    ggplot(df, aes(x="min", y="mean"))
    + geom_point()
    + theme_538()
    + labs(x="kgb", y="cia", title="hello")
    + geom_smooth()
).draw()
