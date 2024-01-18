import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

for dataset in ['wikipedia', 'reddit', 'mooc', 'lastfm', 'myket', 'enron', 'SocialEvo', 'uci', 'Flights', 'CanParl', 'USLegis', 'UNtrade', 'UNvote', 'Contacts']:
  df = pd.read_csv(f"/home/ayush/DyGLib/processed_data/{dataset}/ml_{dataset}.csv")

  x = df["u"].unique().tolist() + df["i"].unique().tolist()
  x = np.unique(x)

  # First seen
  m1 = np.ones(len(x))*(-1)
  for idx, out in enumerate(df[["u", "i"]].to_numpy()):
    u = out[0]
    i = out[1]
    if m1[u-1] == -1:
      m1[u-1] = idx
    if m1[i-1] == -1:
      m1[i-1] = idx

  m2 = np.ones(len(x))*(-1)
  for idx, out in reversed(list(enumerate(df[["u", "i"]].to_numpy()))):
    u = out[0]
    i = out[1]
    if m2[u-1] == -1:
      m2[u-1] = idx
    if m2[i-1] == -1:
      m2[i-1] = idx

  plt.hist(m1, bins=30, color='skyblue', edgecolor='black')
  plt.hist(m2, bins=30, color='red', edgecolor='black')
  plt.savefig(f"/home/ayush/DyGLib/processed_data/stats_images/{dataset}_all_seen.jpg")
  plt.close()
  # Last seen
  # plt.savefig(f"/home/ayush/DyGLib/processed_data/stats_images/{dataset}_last_seen.jpg")
  # plt.close()