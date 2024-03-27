import cta_dataset



dataset = cta_dataset.CTAngiographyNoConditionDataset(root_dir='/home/brody/Laboratory/cta-diffusion/experiments/test_exp/', csv='annotations.csv')

print(dataset.len())
image = dataset.__getitem__(24023)
print(image.shape)
