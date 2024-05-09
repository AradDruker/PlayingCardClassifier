import torch
import torchvision.transforms as transforms

from classes import PlayingCardDataset
from utils import config, file_utils, visualize_predictions

transform = transforms.Compose([
transforms.Resize((128, 128)),
transforms.ToTensor(),
])

class prediction:
    def __init__(self, model, device):
        self.test_dataset = PlayingCardDataset(config['test_folder'], transform=transform)
        self.model = model
        self.device = device
              
    def predict(self, model, image_tensor, device):
        model.eval()
        with torch.no_grad():
            image_tensor = image_tensor.to(device)
            outputs = model(image_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
        return probabilities.cpu().numpy().flatten()

    def test_one_card(self):
        test_image = file_utils.select_random_file_from_parent('C:/Users/aradd/OneDrive/Documents/VS Code/playing_card_classifier/dataset/test')
        original_image, image_tensor = file_utils.preprocess_image(test_image, transform)
        probabilities = self.predict(self.model, image_tensor, self.device)

        class_names = self.test_dataset.classes 
        visualize_predictions(original_image, probabilities, class_names)