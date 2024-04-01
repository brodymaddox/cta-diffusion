import torch
import os
import dataloader
import matplotlib.pyplot as plt
import torch.nn.functional as F
import model
import pandas as pd
from torchvision import transforms
from PIL import Image

# setup sample

experiment_to_sample = 'acas'
model_to_sample = 'acas_15_epoch.pt'
desired_num_samples = 10

# Which test condition are we using?
tc = 0

# Setup image save counter
d = 0

# Are we doing a conditional test
conditional = True # If true, sample conditionallyi

IMG_SIZE = 256

# Move to experiments

os.chdir('..')
os.chdir('experiments')
os.chdir(experiment_to_sample)

# Get path for generated images
os.chdir('gen_images')
gen_image_path = os.getcwd()
os.chdir('..') # Change back

# Store Conditions to Dataframe
test_conditions = pd.read_csv('test_conditions.csv')

# Load Model
model_dict = torch.load(os.path.join(os.getcwd(), model_to_sample))
if conditional:
    unet = model.ChannelConditionalUNET().to('cuda')
else:
    unet = model.SimpleUnet().to('cuda')
unet.load_state_dict(model_dict['state_dict'])

# Code to actualy sample (directly from control)

def linear_beta_schedule(timesteps, start=0.0001, end=0.02):
    return torch.linspace(start, end, timesteps)

def get_index_from_list(vals, t, x_shape):
    """
    returns a specfic index t of a passed list of values
    while considering the batch dimension
    """
    batch_size = t.shape[0]
    out = vals.gather(-1, t.cpu())
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)

def forward_diffusion_sample(x_0, t, device="cpu"):
    """
    takes an image and a timestep as input
    returns noisy version of it
    """
    noise = torch.randn_like(x_0)
    sqrt_alphas_cumprod_t = get_index_from_list(sqrt_alphas_cumprod, t, x_0.shape)
    sqrt_one_minus_alphas_cumprod_t = get_index_from_list(
        sqrt_one_minus_alphas_cumprod, t, x_0.shape
    )
    # mean + variance
    #print((sqrt_alphas_cumprod_t.to(device) * x_0.to(device) + sqrt_one_minus_alphas_cumprod_t.to(device) * noise.to(device)).shape)
    return sqrt_alphas_cumprod_t.to(device) * x_0.to(device) \
    + sqrt_one_minus_alphas_cumprod_t.to(device) * noise.to(device), noise.to(device)

# Define Beta Schedule
T = 300
betas = linear_beta_schedule(timesteps=T)

device = torch.device('cuda' if torch.cuda.is_available() else "cpu")

# Pre-calculate different terms for closed form
alphas = 1. - betas
alphas_cumprod = torch.cumprod(alphas, axis=0)
alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1,0), value=1.0)
sqrt_recip_alphas = torch.sqrt(1.0 / alphas)
sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)
posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

@torch.no_grad()
def sample_timestep(x, t):
    """
    Calls the model to predict the noise in the image and returns
    the denoised image.
    Applies noise to this image, if we are not in the last step yet.
    """
    betas_t = get_index_from_list(betas, t, x.shape)
    sqrt_one_minus_alphas_cumprod_t = get_index_from_list(
        sqrt_one_minus_alphas_cumprod, t, x.shape
    )
    sqrt_recip_alphas_t = get_index_from_list(sqrt_recip_alphas, t, x.shape)

    # Call model (current image - noise prediction)
    model_mean = sqrt_recip_alphas_t * (
        x - betas_t * unet(x, t) / sqrt_one_minus_alphas_cumprod_t
    )
    posterior_variance_t = get_index_from_list(posterior_variance, t, x.shape)

    if t == 0:
        # As pointed out by Luis Pereira (see YouTube comment)
        # The t's are offset from the t's in the paper
        return model_mean
    else:
        noise = torch.randn_like(x)
        return model_mean + torch.sqrt(posterior_variance_t) * noise


@torch.no_grad()
def sample_plot_image():
    # Sample noise
    img_size = IMG_SIZE
    img = torch.randn((1, 1, img_size, img_size), device=device)
    plt.figure(figsize=(15,15))
    plt.axis('off')
    num_images = 10
    stepsize = int(T/num_images)
    
    # Reference global d
    global d

    # Setup image_transform
    image_transform = transforms.Compose([
        transforms.Normalize(mean=[-1.0], std=[2.0]),
        transforms.ToPILImage()

    ])

    for i in range(0,T)[::-1]:
        t = torch.full((1,), i, device=device, dtype=torch.long)
        img = sample_timestep(img, t)
        # Edit: This is to maintain the natural range of the distribution
        img = torch.clamp(img, -1.0, 1.0)
        
        if i == 0: # If we are at the fully denoised image
            img_cp = torch.clone(img) # Copy the Tensor
            pil_image = image_transform(img_cp.squeeze().unsqueeze(0)) # Convert the squeezed tensor (remove batch dim) to a PIL Image
            pil_image.save(os.path.join(gen_image_path, str(d) + model_to_sample + '.png')) # Save image as d counter plus .png to generated images folder
            d = d + 1 # Increment D

        if i % stepsize == 0:
            plt.subplot(1, num_images, int(i/stepsize)+1)
            dataloader.show_tensor_image(img.detach().cpu())
    plt.show()

# Code to sample conditionally    

@torch.no_grad()
def sample_timestep_cond(x, condition, t):
    """
    Calls the model to predict the noise in the image and returns
    the denoised image.
    Applies noise to this image, if we are not in the last step yet.
    """
    betas_t = get_index_from_list(betas, t, x.shape)
    sqrt_one_minus_alphas_cumprod_t = get_index_from_list(
        sqrt_one_minus_alphas_cumprod, t, x.shape
    )
    sqrt_recip_alphas_t = get_index_from_list(sqrt_recip_alphas, t, x.shape)

    # Call model (current image - noise prediction)
    model_mean = sqrt_recip_alphas_t * (
        x - betas_t * unet(x, condition, t) / sqrt_one_minus_alphas_cumprod_t
    )
    posterior_variance_t = get_index_from_list(posterior_variance, t, x.shape)

    if t == 0:
        # As pointed out by Luis Pereira (see YouTube comment)
        # The t's are offset from the t's in the paper
        return model_mean
    else:
        noise = torch.randn_like(x)
        return model_mean + torch.sqrt(posterior_variance_t) * noise

@torch.no_grad()
def sample_plot_image_cond():
    # Sample noise
    img_size = IMG_SIZE
    img = torch.randn((1, 1, img_size, img_size), device=device)
    cond = torch.FloatTensor(test_conditions.loc[tc,:].values.flatten().tolist()) # Load the specified condition as what we are testing
    cond = cond.unsqueeze(0) # Add batch dim of 1
    plt.figure(figsize=(15,15))
    plt.axis('off')
    num_images = 10
    stepsize = int(T/num_images)

    image_transform = transforms.Compose([
        transforms.Normalize(mean=[-1.0], std=[2.0]),
        transforms.ToPILImage()

    ])

    for i in range(0,T)[::-1]:
        t = torch.full((1,), i, device=device, dtype=torch.long)
        img = sample_timestep_cond(img,cond,t)
        # Edit: This is to maintain the natural range of the distribution
        img = torch.clamp(img, -1.0, 1.0)

        if i == 0: # If we are at the fully denoised image
            img_cp = torch.clone(img) # Copy the Tensor
            pil_image = image_transform(img_cp.squeeze().unsqueeze(0)) # Convert the squeezed tensor (remove batch dim) to a PIL Image
            pil_image.save(os.path.join(gen_image_path, str(d) + model_to_sample + '.png')) # Save image as d counter plus .png to generated images folder
            d = d + 1 # Increment D

        if i % stepsize == 0:
            plt.subplot(1, num_images, int(i/stepsize)+1)
            dataloader.show_tensor_image(img.detach().cpu())
    plt.show()


# Conduct the sampling

for i in range(desired_num_samples):
    if conditional:
        sample_plot_image_cond()
    else:
        sample_plot_image()
