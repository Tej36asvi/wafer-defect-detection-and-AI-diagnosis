import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# data loading
print("Loading the semiconductor wafer defect dataset")
path = "./data/LSWMD.pkl"
df = pd.read_pickle(path)
print(df.columns)
print(df.head())
# fix labels
# extract the inner label string
print("Cleaning the defect labels ") 
df['failureType'] = df['failureType'].apply(lambda x: x[0][0] if len(x)>0 else 'none')
print(df.head())

# check counts

print(f"Total wafers loaded: {len(df)}")
print("Defect Counts: ")
print(df['failureType'].value_counts())

# visualization
# plot one sample for each defect type
defects = ['Center', 'Donut', 'Edge-Loc', 'Edge-Ring', 'Loc', 'Random', 'Scratch', 'Near-full']

fig, axes = plt.subplots(2, 4, figsize=(14, 8))
axes = axes.flatten() # flatten axes to one dimension

for i, defect in enumerate(defects):
    # get first wafer for this defect type
    sample = df[df['failureType'] == defect].iloc[0]
    img = sample['waferMap']
    
    # plot image for this defect type
    axes[i].imshow(img, cmap='inferno') # map matrix values to colors
    axes[i].set_title(defect)
    axes[i].axis('off')

plt.tight_layout()
plt.show()