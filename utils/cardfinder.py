import cv2
import numpy as np
import os

from .cardsolver import solver
from .config import config

def order_points(pts):
    """Sort the points based on their x-coordinates and y-coordinates to identify corners"""
    rect = np.zeros((4, 2), dtype="float32")

    # Sum and difference to find top-left (smallest sum) and bottom-right (largest sum)
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]  # Top-left point
    rect[2] = pts[np.argmax(s)]  # Bottom-right point

    # Difference to find top-right (smallest diff) and bottom-left (largest diff)
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # Top-right point
    rect[3] = pts[np.argmax(diff)]  # Bottom-left point

    return rect

def four_point_transform(image, pts):
    """Perform a perspective transform on a region in an image defined by 'pts'."""
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    # Calculate the width and height of the new image
    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    width = max(int(widthA), int(widthB))

    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    height = max(int(heightA), int(heightB))

    # Ensure the longer side is vertical
    if width > height:
        # Card is horizontal, rotate points to make the card vertical
        rect = np.array([tr, br, bl, tl], dtype="float32")
        width, height = height, width  # Swap dimensions

    # Prepare destination points for perspective transformation
    dst = np.array([
        [0, 0],  # Top-left corner
        [width - 1, 0],  # Top-right corner
        [width - 1, height - 1],  # Bottom-right corner
        [0, height - 1]  # Bottom-left corner
    ], dtype="float32")

    # Compute the perspective transform matrix and apply it
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (width, height))

    return warped
def ensure_image_exists_cv(path):
    try:
        # Try to open the image to see if it exists
        img = cv2.imread(path)
        if img is None:
            raise FileNotFoundError
    except FileNotFoundError:
        #print(f"No such file: {path}. Creating a blank image.")
        # Create a blank white image (you can adjust the color and size)
        img = np.ones((128, 128, 3), dtype=np.uint8) * 255  # 255 for white
        # Ensure the directory exists before saving
        os.makedirs(os.path.dirname(path), exist_ok=True)
        cv2.imwrite(path, img)


def find_and_save_white_card():
    solver_instance = solver()
    # Initialize the camera capture
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open camera.")
        return
    ensure_image_exists_cv(config['captured_card'])

    try:
        while True:
            display_text = solver_instance.card_solver()
            # Read a frame from the camera
            ret, frame = cap.read()
            if not ret:
                print("Error: Can't receive frame (stream end?). Exiting ...")
                break

            # Convert frame to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Apply Gaussian blurring
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            
            # Edge detection using Canny
            edges = cv2.Canny(blurred, 50, 150)
            
            # Find contours
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Loop over the contours
            for contour in contours:
                # Approximate the contour
                peri = cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
                
                # If the contour has 4 points, it might be a card
                if len(approx) == 4 and cv2.contourArea(approx) > 1000:
                    # Correct the perspective and extract the card
                    warped = four_point_transform(frame, approx.reshape(4, 2))
                    
                    # Save the corrected image
                    cv2.imwrite(config['captured_card'], warped)
                    
                    # Draw a rectangle for visual feedback (on the live display only)
                    cv2.drawContours(frame, [approx], -1, (0, 255, 0), 3)

            # Use the imported text in the video feed
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(frame, display_text, (30, 50), font, 1, (0, 0, 255), 2, cv2.LINE_AA)

            # Display the resulting frame
            cv2.imshow('Card Detector', frame)
            
            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        # When everything is done, release the capture
        cap.release()
        cv2.destroyAllWindows()

#find_and_save_white_card()