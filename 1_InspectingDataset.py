import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 1. Data loading
print("Loading the semiconductor wafer defect dataset")
path = "./data/LSWMD.pkl"
df = pd.read_pickle(path)
print(df.columns)
print(df.head())
# 2. Fix Labels
# The dataset has labels in array format (e.g., [['none']]), so we extract just the inner string.
print("Cleaning the defect labels ") 
df['failureType'] = df['failureType'].apply(lambda x: x[0][0] if len(x)>0 else 'none')
print(df.head())

# 3. Checking counts

print(f"Total wafers loaded: {len(df)}")
print("Defect Counts: ")
print(df['failureType'].value_counts())

# 4. Visualization
# Plot one example for each of the 8 main defect types
defects = ['Center', 'Donut', 'Edge-Loc', 'Edge-Ring', 'Loc', 'Random', 'Scratch', 'Near-full']

fig, axes = plt.subplots(2, 4, figsize=(14, 8))
axes = axes.flatten() # 1D [0,1,2,3,4,5,6,7] instead of 2D [[0,1,2,3],[4,5,6,7]]

for i, defect in enumerate(defects):
    # Get the first wafer that matches the current defect type
    sample = df[df['failureType'] == defect].iloc[0]
    img = sample['waferMap']
    
    # Plotting image of a unique defect type
    axes[i].imshow(img, cmap='inferno') # converts the numbers in the 2D matrix to colours
    axes[i].set_title(defect)
    axes[i].axis('off')

plt.tight_layout()
plt.show()