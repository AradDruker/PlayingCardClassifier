import torch
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder

import pandas as pd
import numpy as np

from .config import config
from .file_utils import file_utils
from model import prediction, train_model


class solver:
    def __init__(self):
        self.transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        ])

        test_folder = config['test_folder']
        self.target_to_class = {v: k for k, v in ImageFolder(test_folder).class_to_idx.items()}

        filename = config['filename']
        df = pd.read_csv(filename)
        minValue_val_loss_idx = df['Validation Loss'].idxmin()
        best_result_learning_rate, best_result_batch, best_result_epoch = df.iloc[minValue_val_loss_idx, [2, 3, 4]]
        print(f"Best batch: {int(best_result_batch)}, Best learning rate: {best_result_learning_rate}, Best number of epochs: {int(best_result_epoch)}")

        self.model = torch.load(config['trained_model'])
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    def card_solver(self):
        test_image = config['captured_card']
        _, image_tensor = file_utils.preprocess_image(test_image, self.transform)
        #print(image_tensor)
        
        probabilities = prediction(self.model, self.device).predict(self.model, image_tensor, self.device) 
        max_value = np.argmax(probabilities)

        predicted_card = self.target_to_class[max_value]
        #print(predicted_card)
        return predicted_card

