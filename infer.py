import torch, torchaudio
import argparse
from preprocess import process_audio_data, process_image_data
from train import WatermelonModel


def infer(audio, image, model, device):
    # Load and preprocess the input data
    audio, sr = torchaudio.load(audio)
    mfcc = process_audio_data(audio, sr).to(device)
    img = process_image_data(image).to(device)
    if mfcc is None or img is None:
        return None
    
    # Run inference
    with torch.no_grad():
        predicted_sweetness = model(mfcc, img).item()

    return predicted_sweetness


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Watermelon Sweetness Prediction")
    parser.add_argument(
        "--model_path", type=str, required=True, help="Path to the saved model file"
    )
    parser.add_argument(
        "--audio_path", type=str, required=True, help="Path to audio file"
    )
    parser.add_argument(
        "--image_path", type=str, required=True, help="Path to image file"
    )
    args = parser.parse_args()

    # Initialize the model and device
    print(f"\033[92mINFO\033[0m: PyTorch version: {torch.__version__}")
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else "cpu"
    )
    print(f"\033[92mINFO\033[0m: Using device: {device}")
    model = WatermelonModel().to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))

    # Example paths to audio and image files
    audio_path = args.audio_patb
    image_path = args.image_path

    # Run inference
    sweetness = infer(audio_path, image_path, model, device)
    print(f"Predicted sweetness: {sweetness}")
