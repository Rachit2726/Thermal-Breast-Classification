"""
Generate synthetic thermal images using Stable Diffusion img2img pipeline.
"""
import torch
from diffusers import StableDiffusionImg2ImgPipeline
from PIL import Image
import os
from config_loader import load_config


def generate_synthetic_images(cfg=None):
    if cfg is None:
        cfg = load_config()

    gen_cfg = cfg["generation"]
    input_base = cfg["data"]["preprocessed_dir"]
    output_base = cfg["data"]["synthetic_dir"]

    os.makedirs(output_base, exist_ok=True)

    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        torch_dtype=torch.float32,
        safety_checker=None
    ).to(gen_cfg["device"])

    for class_name in cfg["data"]["classes"]:
        prompt = gen_cfg["prompts"][class_name]
        in_dir = os.path.join(input_base, class_name)
        out_dir = os.path.join(output_base, class_name)
        os.makedirs(out_dir, exist_ok=True)

        images = [f for f in os.listdir(in_dir) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
        print(f"{class_name}: {len(images)} images found")

        for idx, img_name in enumerate(images):
            init_image = Image.open(os.path.join(in_dir, img_name)).convert("RGB").resize((256, 256))
            result = pipe(
                prompt=prompt,
                image=init_image,
                strength=gen_cfg["strength"],
                guidance_scale=gen_cfg["guidance_scale"]
            )
            image = result.images[0].convert("L").convert("RGB")
            save_path = os.path.join(out_dir, f"synth_{idx+1:03}.png")
            image.save(save_path)
            print(f"Saved: {save_path}")

    print("✅ Synthetic generation completed")


if __name__ == "__main__":
    generate_synthetic_images()
