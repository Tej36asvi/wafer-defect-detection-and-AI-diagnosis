import pandas as pd
import numpy as np
import cv2
from scipy import ndimage
import random
import os
from sklearn.model_selection import train_test_split
import gc # Garbage Collection to free RAM

# CONFIGURATION
RAW_DATA_PATH = "./data/LSWMD.pkl"
TRAIN_OUTPUT = "./data/train_set.pkl"
TEST_OUTPUT = "./data/test_set.pkl"
IMG_SIZE = 64

# SCALING TARGETS
TARGETS = {
    'TRAIN_SIZE': 8000, 
    'TEST_SIZE': 2000    
}

# HELPERS
def resize_wafer_map(wafer_map):
    return cv2.resize(wafer_map, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_NEAREST)

def augment_image(img):
    strategy = np.random.randint(0, 6)
    if strategy == 0:   return ndimage.rotate(img, 90, reshape=False, order=0)
    elif strategy == 1: return ndimage.rotate(img, 180, reshape=False, order=0)
    elif strategy == 2: return ndimage.rotate(img, 270, reshape=False, order=0)
    elif strategy == 3: return np.fliplr(img)
    elif strategy == 4: return np.flipud(img)
    elif strategy == 5: 
        noise = np.random.normal(0, 0.05, img.shape)
        return np.clip(img + noise, 0, 1)
    return img

def process_partition(df, target_count, partition_name, class_name):
    current_count = len(df)
    if current_count == 0: return []
    final_rows = df.to_dict('records')
    
    if current_count < target_count:
        needed = target_count - current_count
        print(f"   -> {partition_name} ({class_name}): Scaling {current_count} -> {target_count}")
        real_samples = df.to_dict('records')
        for _ in range(needed):
            source_row = random.choice(real_samples)
            img = source_row['waferMap_resized']
            syn_img = augment_image(img)
            new_row = source_row.copy()
            new_row['waferMap_resized'] = syn_img
            new_row['is_synthetic'] = True
            final_rows.append(new_row)
            
    elif current_count > target_count:
        print(f"   -> {partition_name} ({class_name}): Downsampling {current_count} -> {target_count}")
        final_rows = random.sample(final_rows, target_count)
    else:
        print(f"   -> {partition_name} ({class_name}): Exact match")
        
    return final_rows

# MAIN EXECUTION
def main():
    print("1. Loading raw data ")
    if not os.path.exists(RAW_DATA_PATH):
        print(f"Error: {RAW_DATA_PATH} not found.")
        return
    df = pd.read_pickle(RAW_DATA_PATH)

    print("  Cleaning labels ")
    df['failureType'] = df['failureType'].apply(lambda x: x[0][0] if x.size > 0 else 'none')
    valid_failures = ['Center', 'Donut', 'Edge-Loc', 'Edge-Ring', 'Loc', 'Random', 'Scratch', 'Near-full', 'none']
    df = df[df['failureType'].isin(valid_failures)].copy()

    # DROPPING EXCESS 'NONE' 
    print("2. Memory Optimization: Dropping excess 'none' class ")
    

    df_none = df[df['failureType'] == 'none']
    df_others = df[df['failureType'] != 'none']
    
    print(f"   Original 'none' count: {len(df_none)}")
    if len(df_none) > 12000:
        df_none = df_none.sample(n=12000, random_state=42)
    
    # Recombining and deleting old objects to free RAM
    df = pd.concat([df_others, df_none])
    del df_none, df_others
    gc.collect() # Forced RAM cleanup
    
    print(f"   Reduced Dataset Size: {len(df)} (Safe to resize)")

    # 3. RESIZING IMAGES (Now running on only ~40k images instead of 800k)
    print("3. Resizing images ")
    resized_maps = [resize_wafer_map(w) for w in df.waferMap]
    df['waferMap_resized'] = [x / 2.0 for x in resized_maps]
    
    # Dropping raw heavy column
    df = df.drop(columns=['waferMap'])
    gc.collect()

    # 4. SPLITTING REAL DATA
    print("-" * 60)
    print("4. Splitting Real Data (80% Train / 20% Test) ")
    train_df_real, test_df_real = train_test_split(
        df, 
        test_size=0.2, 
        stratify=df['failureType'], 
        random_state=42
    )
    
    # 5. SCALING EACH PARTITION
    print("-" * 60)
    print(f"5. Applying Scaling Targets (Train: {TARGETS['TRAIN_SIZE']}, Test: {TARGETS['TEST_SIZE']})...")
    
    final_train_data = []
    final_test_data = []
    unique_classes = df['failureType'].unique()
    
    for label in unique_classes:
        # TRAINING
        cls_train = train_df_real[train_df_real['failureType'] == label]
        final_train_data.extend(process_partition(cls_train, TARGETS['TRAIN_SIZE'], "TRAIN", label))
        
        # TESTING
        cls_test = test_df_real[test_df_real['failureType'] == label]
        final_test_data.extend(process_partition(cls_test, TARGETS['TEST_SIZE'], "TEST", label))

    # 6. SAVING
    print("-" * 60)
    print("6. Saving Files ")
    df_final_train = pd.DataFrame(final_train_data).sample(frac=1).reset_index(drop=True)
    df_final_test = pd.DataFrame(final_test_data).sample(frac=1).reset_index(drop=True)
    
    df_final_train.to_pickle(TRAIN_OUTPUT)
    df_final_test.to_pickle(TEST_OUTPUT)
    
    print(f" DONE ")
    print(f"   Train Set: {len(df_final_train)} images")
    print(f"   Test Set:  {len(df_final_test)} images")

if __name__ == "__main__":
    main()