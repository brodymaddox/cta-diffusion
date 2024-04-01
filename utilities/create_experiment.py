# Written by Brody Maddox

import os
import numpy as np
import shutil
import pandas as pd
import math
from sklearn import preprocessing

"""  Setup Variables """

# Experiment Parameters
upper_slice = 132 # Max 132
lower_slice = 0 # Min 0
slice_stride = 1 # Min 1 Max (upper-lower)
experiment_name = 'acas'

# Set which columns experiment will be conditioned on
condition_columns = ['slice', 'age', 'race', 'cta_occlusion_site', 'tpa', 'lkw2ct', 'baseline_nihss', 'lvo'] # Options: slice, age, gender, race, cta_occlusion_site, tpa, lkw2ct, baseline_nihss, lvo

# Set list of test prompt condiitons
test_conditions = [
        [0,0,0,0,0,0,0,0] # Test Zeros Condition 
    ]



# Get array of all slices for this experiment
experiment_slices = [x for x in range(lower_slice, upper_slice, slice_stride)]

"""  Setup Experiment Directory Structure """

# Move to cta-diffusion
os.chdir('..')

# Move to experiments
os.chdir('experiments')

# Create Experiment
os.mkdir(experiment_name)

# Enter experiment
os.chdir(experiment_name)

# Store Path
exp_path = os.getcwd()

# Create Img Folder
os.mkdir('images')

# Create Generated Images Folder
os.mkdir('gen_images')

# Move to images folder
os.chdir('images')

# Store Path
img_path = os.getcwd()

# Move to cta-diffusion
os.chdir('../../..')
os.chdir('data')
os.chdir('CTA_By_Slice')

# Setup empty annotations
annotations = []

# Read master conditioning data
df0 = pd.read_csv("/home/brody/Laboratory/cta-diffusion/data/train_info.csv")
df1 = pd.read_csv("/home/brody/Laboratory/cta-diffusion/data/val_info.csv")
masterDf = pd.concat([df0, df1], ignore_index=True, sort=False)

# Copy images from all selected slices into img_path
for slice in experiment_slices:
    path = 'slice_' + str(slice) + '/'
    for file in os.listdir(path):

        # Initialize annotations row and place filename
        annotations_row = []
        annotations_row.append(file)

        # Extract which patient image is of
        patient_id = file[:8]

        # Extract slice
        if 'slice' in condition_columns:
            sl = file[len(file)-7:len(file)-4]
            if sl[1] == '_': sl = sl[2] # Case for slice 0-9
            if sl[0] == '_': sl = sl[1:] # Case for slice 10-99

            # Add slice to annotations_row
            annotations_row.append(int(sl))

        # Get Conditions 
        conditions = masterDf.loc[masterDf['subjId']==patient_id]

        # If condition is in the specified columns, and in dataframe, append value
        for condition in condition_columns:
            try:
                annotations_row.append(conditions.iloc[0][condition])
            except:
                pass
        
        will_add = True # Boolean to track whether or not we should add this image and annotations to our experiment

        # If annotations row contains an empty value, skip this image
        for val in annotations_row[1:]: # Skip the string identifier column
            if math.isnan(val):
                will_add = False

        if will_add:        
             # Add annotations to master annotations
            annotations.append(annotations_row)
            # Copy image to experiment image folder
            shutil.copy(os.path.join(path, file), img_path)

# Create column names for annotations.csv
final_column_names = condition_columns.copy()
final_column_names.insert(0, 'subjId')

# Turn annotations into DataFrame
annotations_df = pd.DataFrame(annotations, columns=final_column_names)

# Turn test conditions into DataFrame
test_conditions_df = pd.DataFrame(test_conditions, columns=condition_columns)

# Normalize annotations.csv
x = annotations_df.values
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x[:,1:])
x_combined = np.hstack((x[:, [0]], x_scaled))
annotations_df_normalized = pd.DataFrame(x_combined, columns=final_column_names)

# Normalize test conditions, write to new DataFrame
tc = test_conditions_df.values
tc_scaled = min_max_scaler.transform(tc)
test_conditions_df_normalized = pd.DataFrame(tc_scaled, columns=condition_columns)


# Write annotations.csv
annotations_df_normalized.to_csv(os.path.join(exp_path, 'annotations.csv'), index=False)

# Write test_conditions.csv
test_conditions_df_normalized.to_csv(os.path.join(exp_path, 'test_conditions.csv'), index=False)



