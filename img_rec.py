import cv2
import os
import numpy as np

def process_images(reference_image_path, folder_path, output_path, threshold_value=50):
    # Create the output directory if it doesn't exist
    os.makedirs(output_path, exist_ok=True)

    # Read the reference image (image without objects)
    reference_image = cv2.imread(reference_image_path, cv2.IMREAD_GRAYSCALE)
    if reference_image is None:
        print(f"Error: Could not read reference image at {reference_image_path}")
        return

    # Iterate over all images in the specified folder
    for filename in os.listdir(folder_path):
        image_path = os.path.join(folder_path, filename)
        
        # Read the current image
        current_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if current_image is None:
            print(f"Warning: Could not read image at {image_path}, skipping.")
            continue
        
        # Compute the absolute difference
        abs_diff = cv2.absdiff(reference_image, current_image)
        blurred_diff = cv2.GaussianBlur(abs_diff, (5, 5), 0)
        
        # Apply a binary threshold to create the mask
        #_, object_mask = cv2.threshold(blurred_diff, threshold_value, 255, cv2.THRESH_BINARY)
        _, otsu_mask = cv2.threshold(blurred_diff, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        adaptive_mask = cv2.adaptiveThreshold(blurred_diff, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
        object_mask = cv2.bitwise_and(otsu_mask, adaptive_mask)
        

        
        # Save debug images
        abs_diff_output_path = os.path.join(output_path, f"absdiff_{filename}")
        mask_output_path = os.path.join(output_path, f"mask_{filename}")
        #cv2.imwrite(abs_diff_output_path, abs_diff)
        cv2.imwrite(mask_output_path, object_mask)
        
        print(f"Processed {filename} - Debug images saved to {output_path}")


def process_images_with_slic(reference_image_path, folder_path, output_path, method="otsu", 
                             threshold_value=50, region_size=40, ruler=30, slic_iterations=10):
    os.makedirs(output_path, exist_ok=True)

    # Read the reference image (image without objects)
    reference_image = cv2.imread(reference_image_path, cv2.IMREAD_GRAYSCALE)
    if reference_image is None:
        print(f"Error: Could not read reference image at {reference_image_path}")
        return

    for filename in os.listdir(folder_path):
        image_path = os.path.join(folder_path, filename)
        current_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        color_image = cv2.imread(image_path)  # Read the color image for SLIC visualization
        if current_image is None:
            print(f"Warning: Could not read image at {image_path}, skipping.")
            continue
        
        # Compute the absolute difference
        abs_diff = cv2.absdiff(reference_image, current_image)
        abs_diff_blur = cv2.GaussianBlur(abs_diff, (5, 5), 0)

        # Apply thresholding
        if method == "adaptive":
            object_mask = cv2.adaptiveThreshold(
                abs_diff_blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
            )
        elif method == "otsu":
            _, object_mask = cv2.threshold(abs_diff_blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        else:  # Fixed threshold
            _, object_mask = cv2.threshold(abs_diff_blur, threshold_value, 255, cv2.THRESH_BINARY)

        # Morphological operations to clean up the mask
        kernel = np.ones((3, 3), np.uint8)
        object_mask = cv2.morphologyEx(object_mask, cv2.MORPH_CLOSE, kernel)

        # Convert the grayscale image to color (required for SLIC)
        next_frame = cv2.cvtColor(current_image, cv2.COLOR_GRAY2BGR)

        # Apply SLIC segmentation
        slic = cv2.ximgproc.createSuperpixelSLIC(next_frame,
                                                 algorithm=cv2.ximgproc.MSLIC,
                                                 region_size=region_size,
                                                 ruler=ruler)
        slic.iterate(slic_iterations)

        # Get the labels and number of segments
        labels = slic.getLabels() + 1  # Labels start at 0, so add 1 for clarity
        num_labels = np.max(labels) + 1

        # Create a mask for the SLIC segments
        slic_mask = np.zeros_like(current_image, dtype=np.uint8)
        slic_mask[slic.getLabelContourMask(thick_line=True) > 0] = 255  # Contour mask for visualization

        # Combine SLIC and object mask
        combined_mask = cv2.bitwise_and(object_mask, slic_mask)

        # Save debug images
        abs_diff_output_path = os.path.join(output_path, f"absdiff_{filename}")
        mask_output_path = os.path.join(output_path, f"mask_{filename}")
        slic_output_path = os.path.join(output_path, f"slic_{filename}")
        combined_mask_output_path = os.path.join(output_path, f"combined_mask_{filename}")

        cv2.imwrite(abs_diff_output_path, abs_diff)
        cv2.imwrite(mask_output_path, object_mask)
        cv2.imwrite(slic_output_path, slic_mask)
        cv2.imwrite(combined_mask_output_path, combined_mask)

        print(f"Processed {filename}: Debug images saved to {output_path}")

# Example usage
reference_image_path = "/home/dzhura/ComputerVision/data/img/reference.JPG"  # Replace with your reference image path
folder_path = "/home/dzhura/ComputerVision/data/img"  # Replace with your folder path
output_path = "/home/dzhura/ComputerVision/data/img/out"  # Replace with your output folder path
threshold_value = 25  # Adjust threshold value as needed

process_images_with_slic(reference_image_path, folder_path, output_path, method="adaptive")
