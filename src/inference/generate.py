import torch
from torchvision import transforms
from models.cnn.deep_cnn import DeepCNN
from models.transformer.transformer_model import TransformerModel
from models.diffusion.diffusion_model import DiffusionModel

def load_model(model_type, model_path):
    if model_type == 'cnn':
        model = DeepCNN()
    elif model_type == 'transformer':
        model = TransformerModel()
    elif model_type == 'diffusion':
        model = DiffusionModel()
    else:
        raise ValueError("Unsupported model type. Choose from 'cnn', 'transformer', or 'diffusion'.")
    
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

def generate_images(model, num_samples, transform=None):
    noise = torch.randn(num_samples, 3, 64, 64)  # Adjust dimensions as needed
    with torch.no_grad():
        generated_images = model(noise)
    if transform:
        generated_images = transform(generated_images)
    return generated_images

def generate_text(model, input_text, num_samples):
    with torch.no_grad():
        generated_text = model.generate(input_text, num_samples)
    return generated_text

def main(model_type, model_path, num_samples, input_text=None):
    model = load_model(model_type, model_path)
    
    if model_type == 'cnn':
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((256, 256)),
            transforms.ToTensor()
        ])
        images = generate_images(model, num_samples, transform)
        # Save or display images as needed
    elif model_type == 'transformer':
        if input_text is None:
            raise ValueError("Input text is required for text generation.")
        text = generate_text(model, input_text, num_samples)
        # Save or display generated text as needed
    elif model_type == 'diffusion':
        # Implement diffusion model generation logic here
        pass

if __name__ == "__main__":
    # Example usage
    main(model_type='cnn', model_path='path/to/cnn_model.pth', num_samples=5)
    # For text generation, use:
    # main(model_type='transformer', model_path='path/to/transformer_model.pth', num_samples=5, input_text='Once upon a time')