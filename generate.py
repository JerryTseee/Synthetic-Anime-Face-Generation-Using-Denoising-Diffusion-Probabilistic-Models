import torch
from diffusers import DDPMScheduler, UNet2DModel
from PIL import Image
import os
from tqdm import tqdm

def generate_images(model, output_dir, num_images=4, image_size=128, epoch=None):
    noise_scheduler = DDPMScheduler(num_train_timesteps=1000)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    # Create epoch-specific directory if epoch is provided
    if epoch is not None:
        output_dir = os.path.join(output_dir, f"epoch_{epoch:04d}")
    os.makedirs(output_dir, exist_ok=True)

    batch_size = min(8, num_images)
    
    with torch.no_grad():
        for batch_idx in tqdm(range(0, num_images, batch_size), desc="Generating images"):
            current_batch_size = min(batch_size, num_images - batch_idx)

            # start from pure noise
            noise = torch.randn(
                current_batch_size, 3, image_size, image_size
            ).to(device)

            # denoising loop
            for t in tqdm(noise_scheduler.timesteps, leave=False, desc="Denoising"):
                # predict noise
                noise_pred = model(noise, t).sample
                # compute previous image
                noise = noise_scheduler.step(noise_pred, t, noise).prev_sample

            # convert to images
            for i in range(current_batch_size):
                img = noise[i].cpu()
                img = (img/2 + 0.5).clamp(0, 1)  # scale to [0, 1]
                img = img.permute(1, 2, 0).numpy()
                img = (img * 255).astype("uint8")

                img_pil = Image.fromarray(img)
                img_pil.save(os.path.join(output_dir, f"generated_{batch_idx + i:02d}.png"))


# pick a model and directly generate images
def generate_final():
    model_path = "./model/model_epoch_198"
    model = UNet2DModel.from_pretrained(model_path)
    output_dir = "./100_generated_images"
    generate_images(model, output_dir, num_images=100, image_size=128)

if __name__ == "__main__":
    generate_final()