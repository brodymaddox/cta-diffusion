# Written by Brody Maddox

import os
import shutil

"""
TODO:
    Need to create annotations.csv
"""

"""  Setup Variables """

# Experiment Parameters
upper_slice = 132 # Max 132
lower_slice = 0 # Min 0
slice_stride = 1
experiment_name = 'test_exp'

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

# Move to images folder
os.chdir('images')

# Store Path
img_path = os.getcwd()

# Move to cta-diffusion
os.chdir('../../..')
os.chdir('data')
os.chdir('CTA_By_Slice')

# Copy images from all selected slices into img_path
for slice in experiment_slices:
    path = 'slice_' + str(slice) + '/'
    for file in os.listdir(path):
        shutil.copy(os.path.join(path, file), img_path)
        



