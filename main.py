import torch
import pandas as pd
import numpy as np
import time

from utils import config, file_utils, visualize_loss
from model import prediction, train_model, train_model_optimal

start_time = time.time()
filename = config['filename']
lr_base = config['learning_rate']

model_classes = ['BasicCNN_v3', 'BasicCNN_v2', 'BasicCNN', 'LeNet']

j = 1 # Number of runs
for model_class in model_classes:
    for i in range(8):
        batch_size = 1 << i  # Shift left by i bits
        train_losses, val_losses, epoch, lr , model, device = train_model(batch_size, lr_base, model_class)
        
        minValue_val_loss = min(val_losses)
        minValue_train_loss_idx = val_losses.index(minValue_val_loss)
        minValue_train_loss = train_losses[minValue_train_loss_idx]
        num_epochs = minValue_train_loss_idx + 1

        file_utils.save_to_csv(minValue_train_loss, minValue_val_loss, lr, batch_size, num_epochs, model_class, filename)  
        print(f"Run {j}: Data appended to {filename}")
        j += 1

# Test the results
df = pd.read_csv(filename)
minValue_val_loss_idx = df['Validation Loss'].idxmin()
best_result_learning_rate, best_result_batch, best_result_epoch, best_model_class = df.iloc[minValue_val_loss_idx, [2, 3, 4, 5]]
print(f"Best batch: {int(best_result_batch)}, Best learning rate: {best_result_learning_rate}, Best number of epochs: {int(best_result_epoch)}")

train_losses, val_losses, *_ = train_model_optimal(int(best_result_batch), lr_base, int(best_result_epoch), best_model_class)
try:
    torch.save(model, config['trained_model'])
    print('NEW MODEL SAVED')
except:
    pass

visualize_loss(train_losses, val_losses)
prediction(model, device).test_one_card() # Based on test dataset

end_time = time.time()
duration = end_time - start_time
hours, remainder = divmod(duration, 3600)
minutes, seconds = divmod(remainder, 60)
print(f"The script ran for {int(hours)} hours, {int(minutes)} minutes, and {seconds:.2f} seconds.")
