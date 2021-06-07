#%%
import pandas as pd
from pandasgui import show
from pkg_resources import normalize_path

popo = pd.read_csv("./data/police.csv")

# *remove the column that only contains missing values
# popo.drop('county_name', axis=1, inplace=True)
popo.dropna(axis=1, how="all", inplace=True)

# *do men or women speed more often?
# (
#     popo
#     .query('violation == "Speeding"')
#     .driver_gender.value_counts(normalize=True)
# )
# * Does gender affact who gets searched during a stop?
(popo.groupby(["driver_gender"]).search_conducted.value_counts(normalize=True))

(popo.search_type.str.contains("Frisk").value_counts(dropna=False, normalize=True))

# *which year has the least number of stops

combined = popo.stop_date.str.cat(popo.stop_time, sep=" ")
popo["stop_datetime"] = pd.to_datetime(combined)
popo.stop_date.str.slice(0, 4).value_counts()
popo.stop_datetime.dt.year.value_counts()

# *how does drug activity change by time of day?

popo.groupby(popo.stop_datetime.dt.hour).drugs_related_stop.count().plot()

# *do most stops occur at night?
popo.stop_datetime.dt.hour.value_counts().sort_index().plot()


# *find the bad data in the stop_duration column and fix
import numpy as np

popo.stop_duration.value_counts()
popo.loc[
    (popo.stop_duration == "1") | (popo.stop_duration == "2"), "stop_duration"
] = np.nan
