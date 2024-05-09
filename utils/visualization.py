import matplotlib.pyplot as plt


def visualize_predictions(original_image, probabilities, class_names):
        fig, axarr = plt.subplots(1, 2, figsize=(14, 7))
    
        # Display image
        axarr[0].imshow(original_image)
        axarr[0].axis("off")
    
        # Display predictions
        axarr[1].barh(class_names, probabilities)
        axarr[1].set_xlabel("Probability")
        axarr[1].set_title("Class Predictions")
        axarr[1].set_xlim(0, 1)

        plt.tight_layout()
        plt.show()
        
def visualize_loss(train_losses, val_losses):
        plt.plot(train_losses, label='Training Loss')
        plt.plot(val_losses, label='Validation Loss')
        plt.legend()  # Add a legend to distinguish the lines
        plt.title('Training and Validation Losses')  # Optional: add a title
        plt.xlabel('Epoch')  # Optional: label the x-axis
        plt.ylabel('Loss')  # Optional: label the y-axis
        plt.show()
