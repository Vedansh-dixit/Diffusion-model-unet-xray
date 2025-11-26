import torch
import torch.nn as nn
import gradio as gr
from PIL import Image
import numpy as np
import math
import os
from threading import Event
import traceback
import cv2

# Constants
IMG_SIZE = 128
TIMESTEPS = 300
NUM_CLASSES = 2

# Global Cancellation Flag
cancel_event = Event()

# Device Configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Model Definitions ---
class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        half_dim = dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim) * -emb)
        self.register_buffer('embeddings', emb)

    def forward(self, time):
        device = time.device
        embeddings = self.embeddings.to(device)
        embeddings = time[:, None] * embeddings[None, :]
        return torch.cat([embeddings.sin(), embeddings.cos()], dim=-1)

class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, num_classes=2, time_dim=256):
        super().__init__()
        self.num_classes = num_classes
        self.label_embedding = nn.Embedding(num_classes, time_dim)

        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.ReLU(),
            nn.Linear(time_dim, time_dim)
        )

        # Encoder
        self.inc = self.double_conv(in_channels, 64)
        self.down1 = self.down(64 + time_dim * 2, 128)
        self.down2 = self.down(128 + time_dim * 2, 256)
        self.down3 = self.down(256 + time_dim * 2, 512)

        # Bottleneck
        self.bottleneck = self.double_conv(512 + time_dim * 2, 1024)

        # Decoder
        self.up1 = nn.ConvTranspose2d(1024, 256, kernel_size=2, stride=2)
        self.upconv1 = self.double_conv(256 + 256 + time_dim * 2, 256)

        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.upconv2 = self.double_conv(128 + 128 + time_dim * 2, 128)

        self.up3 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.upconv3 = self.double_conv(64 + 64 + time_dim * 2, 64)

        self.outc = nn.Conv2d(64, out_channels, kernel_size=1)

    def double_conv(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def down(self, in_channels, out_channels):
        return nn.Sequential(
            nn.MaxPool2d(2),
            self.double_conv(in_channels, out_channels)
        )

    def forward(self, x, labels, time):
        label_indices = torch.argmax(labels, dim=1)
        label_emb = self.label_embedding(label_indices)
        t_emb = self.time_mlp(time)

        combined_emb = torch.cat([t_emb, label_emb], dim=1)
        combined_emb = combined_emb.unsqueeze(-1).unsqueeze(-1)

        x1 = self.inc(x)
        x1_cat = torch.cat([x1, combined_emb.repeat(1, 1, x1.shape[-2], x1.shape[-1])], dim=1)

        x2 = self.down1(x1_cat)
        x2_cat = torch.cat([x2, combined_emb.repeat(1, 1, x2.shape[-2], x2.shape[-1])], dim=1)

        x3 = self.down2(x2_cat)
        x3_cat = torch.cat([x3, combined_emb.repeat(1, 1, x3.shape[-2], x3.shape[-1])], dim=1)

        x4 = self.down3(x3_cat)
        x4_cat = torch.cat([x4, combined_emb.repeat(1, 1, x4.shape[-2], x4.shape[-1])], dim=1)

        x5 = self.bottleneck(x4_cat)

        x = self.up1(x5)
        x = torch.cat([x, x3], dim=1)
        x = torch.cat([x, combined_emb.repeat(1, 1, x.shape[-2], x.shape[-1])], dim=1)
        x = self.upconv1(x)

        x = self.up2(x)
        x = torch.cat([x, x2], dim=1)
        x = torch.cat([x, combined_emb.repeat(1, 1, x.shape[-2], x.shape[-1])], dim=1)
        x = self.upconv2(x)

        x = self.up3(x)
        x = torch.cat([x, x1], dim=1)
        x = torch.cat([x, combined_emb.repeat(1, 1, x.shape[-2], x.shape[-1])], dim=1)
        x = self.upconv3(x)

        return self.outc(x)

class DiffusionModel(nn.Module):
    def __init__(self, model, timesteps=TIMESTEPS, time_dim=256):
        super().__init__()
        self.model = model
        self.timesteps = timesteps
        self.time_dim = time_dim

        # Linear beta schedule with scaling
        scale = 1000 / timesteps
        beta_start = scale * 0.0001
        beta_end = scale * 0.02
        self.betas = torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float64)
        self.alphas = 1. - self.betas
        self.register_buffer('alpha_bars', torch.cumprod(self.alphas, dim=0).float())

    def forward_diffusion(self, x_0, t, noise):
        x_0 = x_0.float()
        noise = noise.float()
        alpha_bar_t = self.alpha_bars[t].view(-1, 1, 1, 1)
        x_t = torch.sqrt(alpha_bar_t) * x_0 + torch.sqrt(1. - alpha_bar_t) * noise
        return x_t

    def forward(self, x_0, labels):
        t = torch.randint(0, self.timesteps, (x_0.shape[0],), device=x_0.device).long()
        noise = torch.randn_like(x_0)
        x_t = self.forward_diffusion(x_0, t, noise)
        predicted_noise = self.model(x_t, labels, t.float())
        return predicted_noise, noise, t

    @torch.no_grad()
    def sample(self, num_images, img_size, num_classes, labels, device, progress_callback=None):
        # Start with random noise
        x_t = torch.randn(num_images, 3, img_size, img_size).to(device)

        # Label handling (one-hot if needed)
        if labels.ndim == 1:
            labels_one_hot = torch.zeros(num_images, num_classes).to(device)
            labels_one_hot[torch.arange(num_images), labels] = 1
            labels = labels_one_hot
        else:
            labels = labels.to(device)

        # ---- REVERTED SAMPLING LOOP WITH NOISE REDUCTION ----
        for t in reversed(range(self.timesteps)):
            if cancel_event.is_set():
                return None
                
            t_tensor = torch.full((num_images,), t, device=device, dtype=torch.float)
            predicted_noise = self.model(x_t, labels, t_tensor)

            # Calculate coefficients
            beta_t = self.betas[t].to(device)
            alpha_t = self.alphas[t].to(device)
            alpha_bar_t = self.alpha_bars[t].to(device)

            mean = (1 / torch.sqrt(alpha_t)) * (x_t - (beta_t / torch.sqrt(1 - alpha_bar_t)) * predicted_noise)
            variance = beta_t

            # Reduced noise injection with lower multiplier
            if t > 0:
                noise = torch.randn_like(x_t) * 0.8  # Reduced noise by 20%
            else:
                noise = torch.zeros_like(x_t)

            x_t = mean + torch.sqrt(variance) * noise
            
            if progress_callback:
                progress_callback((self.timesteps - t) / self.timesteps)

        # Clamp and denormalize
        x_0 = torch.clamp(x_t, -1., 1.)
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)
        x_0 = std * x_0 + mean
        x_0 = torch.clamp(x_0, 0., 1.)

        # ---- ENHANCED SHARPENING ----
        # First apply mild bilateral filtering to reduce noise while preserving edges
        x_np = x_0.cpu().permute(0, 2, 3, 1).numpy()
        filtered = []
        for img in x_np:
            img = (img * 255).astype(np.uint8)
            filtered_img = cv2.bilateralFilter(img, d=5, sigmaColor=15, sigmaSpace=15)
            filtered.append(filtered_img / 255.0)
        x_0 = torch.tensor(np.array(filtered), device=device).permute(0, 3, 1, 2)

        # Then apply stronger unsharp masking
        kernel = torch.ones(3, 1, 5, 5, device=device) / 75
        kernel = kernel.to(x_0.dtype)
        blurred = torch.nn.functional.conv2d(
            x_0,
            kernel,
            padding=2,
            groups=3
        )
        x_0 = torch.clamp(1.5 * x_0 - 0.5 * blurred, 0., 1.)  # Increased sharpening factor

        return x_0

def load_model(model_path, device):
    unet_model = UNet(num_classes=NUM_CLASSES).to(device)
    diffusion_model = DiffusionModel(unet_model, timesteps=TIMESTEPS).to(device)

    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location=device)
        
        if 'model_state_dict' in checkpoint:
            # Handle training checkpoint format
            state_dict = {
                k[6:]: v for k, v in checkpoint['model_state_dict'].items()
                if k.startswith('model.')
            }
            
            # Load UNet weights
            unet_model.load_state_dict(state_dict, strict=False)
            
            # Initialize diffusion model with loaded UNet
            diffusion_model = DiffusionModel(unet_model, timesteps=TIMESTEPS).to(device)
            
            print(f"Loaded UNet weights from {model_path}")
        else:
            # Handle direct model weights format
            try:
                # First try loading full DiffusionModel
                diffusion_model.load_state_dict(checkpoint)
                print(f"Loaded full DiffusionModel from {model_path}")
            except RuntimeError:
                # If that fails, load just the UNet weights
                unet_model.load_state_dict(checkpoint, strict=False)
                diffusion_model = DiffusionModel(unet_model, timesteps=TIMESTEPS).to(device)
                print(f"Loaded UNet weights only from {model_path}")
    else:
        print(f"Weights file not found at {model_path}")
        print("Using randomly initialized weights")

    diffusion_model.eval()
    return diffusion_model

def cancel_generation():
    cancel_event.set()
    return "Generation cancelled"

def generate_images(label_str, num_images, progress=gr.Progress()):
    global loaded_model
    cancel_event.clear()
    
    if num_images < 1 or num_images > 10:
        raise gr.Error("Number of images must be between 1 and 10")
    
    label_map = {'Pneumonia': 0, 'Pneumothorax': 1}
    if label_str not in label_map:
        raise gr.Error("Invalid condition selected")

    labels = torch.zeros(num_images, NUM_CLASSES)
    labels[:, label_map[label_str]] = 1

    try:
        def progress_callback(progress_val):
            progress(progress_val, desc="Generating...")
            if cancel_event.is_set():
                raise gr.Error("Generation was cancelled by user")

        with torch.no_grad():
            images = loaded_model.sample(
                num_images=num_images,
                img_size=IMG_SIZE,
                num_classes=NUM_CLASSES,
                labels=labels,
                device=device,
                progress_callback=progress_callback
            )
        
        if images is None:
            return None, None
            
        processed_images = []
        for img in images:
            img_np = img.cpu().permute(1, 2, 0).numpy()
            img_np = (img_np * 255).clip(0, 255).astype(np.uint8)
            pil_img = Image.fromarray(img_np)
            processed_images.append(pil_img)
        
        if num_images == 1:
            return processed_images[0], processed_images
        else:
            return None, processed_images

    except Exception as e:
        traceback.print_exc()
        raise gr.Error(f"Generation failed: {str(e)}")
    finally:
        torch.cuda.empty_cache()

# Load model
MODEL_NAME = "model_weights.pth"
model_path = MODEL_NAME
print("Loading model...")
try:
    loaded_model = load_model(model_path, device)
    print("Model loaded successfully!")
except Exception as e:
    print(f"Failed to load model: {e}")
    print("Creating dummy model for demonstration")
    loaded_model = DiffusionModel(UNet(num_classes=NUM_CLASSES), timesteps=TIMESTEPS).to(device)

# Gradio UI
with gr.Blocks(theme=gr.themes.Soft(
    primary_hue="violet",
    neutral_hue="slate",
    font=[gr.themes.GoogleFont("Poppins")],
    text_size="md"
)) as demo:
    gr.Markdown("""
    <center>
    <h1>Synthetic X-ray Generator</h1>
    <p><em>Generate synthetic chest X-rays conditioned on pathology</em></p>
    </center>
    """)
    
    with gr.Row():
        with gr.Column(scale=1):
            condition = gr.Dropdown(
                ["Pneumonia", "Pneumothorax"],
                label="Select Condition",
                value="Pneumonia",
                interactive=True
            )
            num_images = gr.Slider(
                1, 10, value=1, step=1,
                label="Number of Images",
                interactive=True
            )
            
            with gr.Row():
                submit_btn = gr.Button("Generate", variant="primary")
                cancel_btn = gr.Button("Cancel", variant="stop")
            
            gr.Markdown("""
            <div style="text-align: center; margin-top: 10px;">
                <small>Note: Generation may take several seconds per image</small>
            </div>
            """)
        
        with gr.Column(scale=2):
            with gr.Tabs():
                with gr.TabItem("Output", id="output_tab"):
                    single_image = gr.Image(
                        label="Generated X-ray",
                        height=400,
                        visible=True
                    )
                    gallery = gr.Gallery(
                        label="Generated X-rays",
                        columns=3,
                        height="auto",
                        object_fit="contain",
                        visible=False
                    )
    
    def update_ui_based_on_count(num_images):
        if num_images == 1:
            return {
                single_image: gr.update(visible=True),
                gallery: gr.update(visible=False)
            }
        else:
            return {
                single_image: gr.update(visible=False),
                gallery: gr.update(visible=True)
            }
    
    num_images.change(
        fn=update_ui_based_on_count,
        inputs=num_images,
        outputs=[single_image, gallery]
    )
    
    submit_btn.click(
        fn=generate_images,
        inputs=[condition, num_images],
        outputs=[single_image, gallery]
    )
    
    cancel_btn.click(
        fn=cancel_generation,
        outputs=None
    )

    demo.css = """
    .gradio-container {
        background: linear-gradient(135deg, #f5f7fa 0%, #e4e8f0 100%);
    }
    .gallery-container {
        background-color: white !important;
    }
    """

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860,share= True)