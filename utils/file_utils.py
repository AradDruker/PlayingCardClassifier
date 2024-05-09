import os
import random
import pandas as pd
from PIL import Image

class file_utils:
    # Load and preprocess the image
    def preprocess_image(image_path, transform):
        image = Image.open(image_path).convert("RGB")
        return image, transform(image).unsqueeze(0)
    
    def select_random_file_from_parent(parent_folder):
        # Check if the provided path is a directory
        if not os.path.isdir(parent_folder):
            print("Error: Provided path is not a directory.")
            return None

        # Get a list of all subfolders in the parent directory
        subfolders = [os.path.join(parent_folder, name) for name in os.listdir(parent_folder)
                    if os.path.isdir(os.path.join(parent_folder, name))]

        # Check if there are any subfolders in the parent directory
        if not subfolders:
            print("Error: No subfolders found in the parent directory.")
            return None

        # Select a random subfolder from the list
        random_subfolder = random.choice(subfolders)

        # Get a list of all files in the randomly selected subfolder
        files_in_subfolder = [file for file in os.listdir(random_subfolder)
                            if os.path.isfile(os.path.join(random_subfolder, file))]

        # Check if there are any files in the randomly selected subfolder
        if not files_in_subfolder:
            print("Error: No files found in the randomly selected subfolder.")
            return None

        # Select a random file from the list of files in the randomly selected subfolder
        random_file = random.choice(files_in_subfolder)

        # Construct the full path to the randomly selected file
        random_file_path = os.path.join(random_subfolder, random_file)

        return random_file_path

    def save_to_csv(final_train_loss, final_val_loss, learning_rate, batch_size, num_epochs, model_class, filename):
        # Create a DataFrame from the losses
        df = pd.DataFrame({
            'Train Loss'     :  [final_train_loss],
            'Validation Loss':  [final_val_loss],
            'learning_rate'  :  [learning_rate],
            'batch_size'     :  [batch_size],
            'num_epochs'     :  [num_epochs],
            'model_class'    :  [model_class]
        })
    
        # Check if the file exists and if so, avoid writing the header again
        if os.path.exists(filename):
            df.to_csv(filename, mode='a', header=False, index=False)
        else:
            df.to_csv(filename, mode='w', header=True, index=False)

