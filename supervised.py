#%%
import pandas as pd
from pandasgui import show
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import statsmodels.formula.api as smf
import numpy as np
import warnings

warnings.filterwarnings("ignore")
font_size = 12
df = pd.merge(
    (
        pd.read_csv("./data/alcohol.csv")
        .loc[:, ["name", "Country Code", "2018"]]
        .rename(columns={"2018": "alcohol"})
    ),
    (
        pd.read_csv("./data/gdp.csv")
        .loc[:, ["name", "Country Code", "2018"]]
        .rename(columns={"2018": "gdp"})
    ),
).merge(
    (
        pd.read_csv("./data/life_exp.csv")
        .loc[:, ["name", "Country Code", "2018"]]
        .rename(columns={"2018": "life"})
    )
)
x_surf, y_surf = np.meshgrid(
    np.linspace(df.alcohol.min(), df.alcohol.max()),
    np.linspace(df.gdp.min(), df.gdp.max()),
)
fig = plt.figure()
ax = plt.figure().add_subplot(111, projection="3d")
ax.set_xlabel("alcohol", fontsize=font_size)
ax.set_ylabel("GDP", fontsize=font_size)
ax.set_zlabel("life expectancy", fontsize=font_size)
ax.scatter(df.alcohol, df.gdp, df.life, c="red", marker="+")
ax.plot_surface(
    x_surf,
    y_surf,
    np.array(
        smf.ols(formula="life ~ alcohol + gdp", data=df)
        .fit()
        .predict(exog=pd.DataFrame({"alcohol": x_surf.ravel(), "gdp": y_surf.ravel()}))
    ).reshape(x_surf.shape),
    alpha=0.1,
)
plt.show()
