import os
import inspect
import fire
import glob
import numpy as np
import torch
from torch import autocast
from PIL import Image
import librosa
import ffmpeg
from librosa import onset
from librosa.feature import melspectrogram
import matplotlib.pyplot as plt

from diffusers import StableDiffusionPipeline
from diffusers.schedulers import LMSDiscreteScheduler

@torch.no_grad()
def diffuse(
        pipe,
        cond_embeddings, 
        cond_latents,  
        num_inference_steps,
        guidance_scale,
        eta,
    ):
    torch_device = torch.device("cpu")

    max_length = cond_embeddings.shape[1] 
    uncond_input = pipe.tokenizer([""], padding="max_length", max_length=max_length, return_tensors="pt")
    uncond_embeddings = pipe.text_encoder(uncond_input.input_ids.to(torch_device))[0]
    text_embeddings = torch.cat([uncond_embeddings, cond_embeddings])

    if isinstance(pipe.scheduler, LMSDiscreteScheduler):
        cond_latents = cond_latents * pipe.scheduler.sigmas[0]

    accepts_offset = "offset" in set(inspect.signature(pipe.scheduler.set_timesteps).parameters.keys())
    extra_set_kwargs = {}
    if accepts_offset:
        extra_set_kwargs["offset"] = 1
    pipe.scheduler.set_timesteps(num_inference_steps, **extra_set_kwargs)

    accepts_eta = "eta" in set(inspect.signature(pipe.scheduler.step).parameters.keys())
    extra_step_kwargs = {}
    if accepts_eta:
        extra_step_kwargs["eta"] = eta

    for i, t in enumerate(pipe.scheduler.timesteps):
        latent_model_input = torch.cat([cond_latents] * 2)
        if isinstance(pipe.scheduler, LMSDiscreteScheduler):
            sigma = pipe.scheduler.sigmas[i]
            latent_model_input = latent_model_input / ((sigma**2 + 1) ** 0.5)

        noise_pred = pipe.unet(latent_model_input, t, encoder_hidden_states=text_embeddings)["sample"]

        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

        if isinstance(pipe.scheduler, LMSDiscreteScheduler):
            cond_latents = pipe.scheduler.step(noise_pred, i, cond_latents, **extra_step_kwargs)["prev_sample"]
        else:
            cond_latents = pipe.scheduler.step(noise_pred, t, cond_latents, **extra_step_kwargs)["prev_sample"]

    cond_latents = 1 / 0.18215 * cond_latents
    image = pipe.vae.decode(cond_latents)
    image = image.tensor
    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.cpu().permute(0, 2, 3, 1).numpy()
    image = (image[0] * 255).astype(np.uint8)

    plt.imshow(image)
    plt.show()

    return image

def interpolate(image_a, image_b, steps):
    ratio = torch.linspace(0, 1, steps)
    result = []
    for r in ratio:
        img = r * image_a + (1 - r) * image_b
        result.append(img)
    return result

def get_audio_features(audio_input):
    # Load the audio file and its sample rate
    audio_data, sample_rate = librosa.load(audio_input)
    print("Audio file path:", audio_input)


    # Compute the mel spectrogram
    n_mels = 80
    n_fft = 2048
    hop_length = 512
    mel_spectrogram = librosa.feature.melspectrogram(
        audio_data, sr=sample_rate, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels
    )

    # Convert to log-mel spectrogram
    log_mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)

    # Compute the beat track
    onset_envelope = onset.onset_strength(y=audio_data, sr=sample_rate)
    bpm, _ = librosa.beat.beat_track(onset_envelope=onset_envelope, sr=sample_rate)

    return log_mel_spectrogram, sample_rate, bpm

def main(
        audio_input,
        prompts = ["abstract geometry", "cosmic landscape"], 
        seeds=[243, 523],
        name = 'ekarma_video_output',
        rootdir = './dreams',
        num_steps = 72,  
        num_inference_steps = 50,
        guidance_scale = 7.5,
        eta = 0.0,
        width = 512,
        height = 512,
):
    print("Number of prompts:", len(prompts))
    print("Number of seeds:", len(seeds))
    print("Prompts:")
    for prompt in prompts:
        print(prompt)
    print("Seeds:")
    for seed in seeds:
        print(seed)
    assert len(prompts) == len(seeds)
    assert height % 8 == 0 and width % 8 == 0

    outdir = os.path.join(rootdir, name)
    os.makedirs(outdir, exist_ok=True)

    # Initializing pipe and setting the torch_device to "cpu"
    pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-3", use_auth_token=True)
    torch_device = torch.device("cpu")
    pipe.unet.to(torch_device)
    pipe.vae.to(torch_device)
    pipe.text_encoder.to(torch_device)

    prompt_embeddings = []
    for prompt in prompts:   
        text_input = pipe.tokenizer(
            prompt,
            padding="max_length",
            max_length=pipe.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt"
        )
        with torch.no_grad():
            embed = pipe.text_encoder(text_input.input_ids.to(torch_device))[0]

        prompt_embeddings.append(embed)
    
    prompt_embedding_a, *prompt_embeddings = prompt_embeddings
    init_seed, *seeds = seeds
    init_a = torch.randn(
        (1, pipe.unet.in_channels, height // 8, width // 8),
        device=torch_device,
        generator=torch.Generator().manual_seed(init_seed)
    )
    
    # Get the audio features
    log_mel_spectrogram, sample_rate, bpm = get_audio_features(audio_input)

    # Compute the duration of each image frame based on the audio beat
    frame_duration = 60.0 / bpm
    frame_hop_duration = frame_duration / num_steps

    # Convert to image
    image_a = diffuse(
        pipe,
        prompt_embedding_a,
        init_a,  
        num_inference_steps,
        guidance_scale,
        eta,
    )
    images = [image_a]

    for prompt_embedding, seed in zip(prompt_embeddings, seeds):
        init_b = torch.randn(
            (1, pipe.unet.in_channels, height // 8, width // 8),
            device=torch_device,
            generator=torch.Generator().manual_seed(seed)
        )
        image_b = diffuse(
            pipe,
            prompt_embedding,
            init_b,  
            num_inference_steps,
            guidance_scale,
            eta,
        )

        images.extend(interpolate(image_a, image_b, num_steps))
        images.append(image_b)
        image_a = image_b

    # Save frames as PNGs
    for i, img in enumerate(images):
        img = Image.fromarray(img)
        img.save(os.path.join(outdir, f'frame_{i:04}.png'))

    # Convert PNG frames to video
    (
        ffmpeg
        .input(os.path.join(outdir, 'frame_*.png'), pattern_type='glob', framerate=25)
        .output(os.path.join(outdir, 'video.mp4'))
        .run()
    )

if __name__ == '__main__':
    audio_input = input("Please enter the audio file path: ")
    fire.Fire(main, audio_input)
