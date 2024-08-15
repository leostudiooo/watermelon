import torch, torchaudio, torchvision
import os
import gradio as gr

from preprocess import image_preprocessing, audio_preprocessing
from train import WatermelonModel
from infer import infer

def load_model(model_path):
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    )
    print(f"\033[92mINFO\033[0m: Using device: {device}")

    model = WatermelonModel().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print(f"\033[92mINFO\033[0m: Loaded model from {model_path}")
    return model

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Watermelon sweetness predictor")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained model")
    args = parser.parse_args()

    model = load_model(args.model_path)

    def predict(audio, image):
        mfcc = audio_preprocessing(audio)
        img = image_preprocessing(image)
        if mfcc is not None and img is not None:
            sweetness = infer(mfcc, img)
            return sweetness.item()
        return None

    audio_input = gr.Audio(label="Upload or Record Audio")
    image_input = gr.Image(label="Upload or Capture Image")
    output = gr.Textbox(label="Predicted Sweetness")

    interface = gr.Interface(
        fn=predict,
        inputs=[audio_input, image_input],
        outputs=output,
        title="Watermelon Sweetness Predictor",
        description="Upload an audio file and an image to predict the sweetness of a watermelon."
    )

    interface.launch()
