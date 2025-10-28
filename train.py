import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.datasets import ImageFolder
from diffusers import DDPMScheduler, UNet2DModel
from PIL import Image
from generate import generate_images
import os
from tqdm import tqdm

# Custom dataset for flat directory structure
class FlatImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        # Get all image files
        self.image_files = [f for f in os.listdir(root_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        print(f"Found {len(self.image_files)} images in {root_dir}")
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.image_files[idx])
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        return image, 0  # Return 0 as dummy label

# configuration
config = {
    "dataset_path": "./dataset/images",
    "image_size": 128,
    "batch_size": 4,
    "num_epochs": 200,
    "lr": 1e-4,
    "output_dir": "./model",
    "generated_images_dir": "./generated_images"
}

# dataset setup
transform = transforms.Compose([
    transforms.Resize((config["image_size"], config["image_size"])),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])


dataset = FlatImageDataset(root_dir=config["dataset_path"], transform=transform)
dataloader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=True, num_workers=4)

# model setup
model = UNet2DModel(
    sample_size=config["image_size"], # height, width
    in_channels=3,
    out_channels=3,
    layers_per_block=2, # num of conv layers in each u-net block
    block_out_channels=(128, 256, 512, 512),
    down_block_types=( # downsampling -> make images smaller
        "DownBlock2D",
        "DownBlock2D",
        "AttnDownBlock2D",
        "AttnDownBlock2D"
    ),
    up_block_types=( # upsampling -> make images larger
        "AttnUpBlock2D",
        "AttnUpBlock2D",
        "UpBlock2D",
        "UpBlock2D"
    )
)


noise_scheduler = DDPMScheduler(num_train_timesteps=1000)
optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"])
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# training loop
model.train()
for epoch in range(config["num_epochs"]):
    total_loss = 0


    for batch, _ in tqdm(dataloader, desc=f"Epoch {epoch+1}/{config['num_epochs']}"):
        clean_images = batch.to(device)

        noise = torch.randn(clean_images.shape).to(device)
        bs = clean_images.shape[0]

        timesteps = torch.randint(0, noise_scheduler.num_train_timesteps, (bs,), device=device).long()

        # add noise to the clean images
        noisy_images = noise_scheduler.add_noise(clean_images, noise, timesteps)

        # predict the noise
        noise_pred = model(noisy_images, timesteps).sample

        loss = F.mse_loss(noise_pred, noise)

        # backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    avg_loss = total_loss / len(dataloader)
    print(f"Epoch {epoch+1} completed. Average Loss: {avg_loss:.4f}")

    # generate some images to see at each epoch end
    generate_images(
        model,
        output_dir=config["generated_images_dir"],
        num_images=4,
        image_size=config["image_size"],
        epoch=epoch+1
    )

    os.makedirs(config["output_dir"], exist_ok=True)
    model.save_pretrained(os.path.join(config["output_dir"], f"model_epoch_{epoch+1}"))
        
# final model save
os.makedirs(config["output_dir"], exist_ok=True)
model.save_pretrained(os.path.join(config["output_dir"], "final_model"))
print("Training completed and model saved.")
