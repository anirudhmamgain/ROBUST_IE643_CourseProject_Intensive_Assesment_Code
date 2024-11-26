
import os
import pydicom
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import cv2
from skimage.metrics import structural_similarity as ssim
import random
import torch.nn.functional as F
import matplotlib.cm as cm
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
import matplotlib.image as mpimg
import nibabel as nib

import os
import pydicom
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import cv2
from skimage.metrics import structural_similarity as ssim
import random
import torch.nn.functional as F
import matplotlib.cm as cm
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
import matplotlib.image as mpimg
import nibabel as nib

import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt

import numpy as np
import matplotlib.pyplot as plt
import torch
import cv2
from scipy.signal import find_peaks
from matplotlib import cm






def liver_demo(nii_file,label_input):
    
    class AttentionGate(nn.Module):
        def __init__(self, F_g, F_l, F_int):
            super(AttentionGate, self).__init__()
            self.W_g = nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0)
            self.W_x = nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0)
            self.psi = nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0)
            self.relu = nn.ReLU(inplace=True)
            self.sigmoid = nn.Sigmoid()

        def forward(self, g, x):
            g1 = self.W_g(g)
            x1 = self.W_x(x)
            psi = self.relu(g1 + x1)
            psi = self.sigmoid(self.psi(psi))
            return x * psi



    class UNet2D(nn.Module):
        def __init__(self, in_channels, out_channels):
            super(UNet2D, self).__init__()
            self.encoder1 = self.conv_block(in_channels, 64)
            self.encoder2 = self.conv_block(64, 128)
            self.encoder3 = self.conv_block(128, 256)
            self.encoder4 = self.conv_block(256, 512)
            
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
            
            self.decoder1 = self.upconv(512, 256)
            self.decoder1_conv = self.conv_block(512, 256)
            
            self.decoder2 = self.upconv(256, 128)
            self.decoder2_conv = self.conv_block(256, 128)
            
            self.decoder3 = self.upconv(128, 64)
            self.decoder3_conv = self.conv_block(128, 64)
            
            self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)
            
            self.att1 = AttentionGate(256, 256, 128)
            self.att2 = AttentionGate(128, 128, 64)
            self.att3 = AttentionGate(64, 64, 32)

        def conv_block(self, in_channels, out_channels):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )

        def upconv(self, in_channels, out_channels):
            return nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)

        def forward(self, x):
            enc1 = self.encoder1(x)
            enc2 = self.encoder2(self.pool(enc1))
            enc3 = self.encoder3(self.pool(enc2))
            enc4 = self.encoder4(self.pool(enc3))
            
            dec1 = self.decoder1(enc4)
            att1 = self.att1(dec1, enc3)
            dec1 = torch.cat((att1, enc3), dim=1)
            dec1 = self.decoder1_conv(dec1)

            dec2 = self.decoder2(dec1)
            att2 = self.att2(dec2, enc2)
            dec2 = torch.cat((att2, enc2), dim=1)
            dec2 = self.decoder2_conv(dec2)

            dec3 = self.decoder3(dec2)
            att3 = self.att3(dec3, enc1)
            dec3 = torch.cat((att3, enc1), dim=1)
            dec3 = self.decoder3_conv(dec3)
            
            return self.final_conv(dec3)    



    class WNet(nn.Module):
        def __init__(self, in_channels, out_channels, num_classes):
            super(WNet, self).__init__()
            self.unet1 = UNet2D(in_channels, out_channels)
            self.fc_conv = nn.Conv2d(out_channels, num_classes, kernel_size=1, stride=1)
            self.unet2 = UNet2D(num_classes, in_channels)

        def forward(self, x):
            seg_output = self.unet1(x)
            fc_output = self.fc_conv(seg_output)
            softmax_output = F.softmax(fc_output, dim=1)
            reconstruction_output = self.unet2(softmax_output)
            return seg_output, softmax_output, reconstruction_output



    def soft_ncuts_loss(output, weight_matrix):
        clusters = output.shape[1]
        P = torch.softmax(output, dim=1)
        cut_value = 0

        for k in range(clusters):
            P_k = P[:, k, :, :]
            cut_value += torch.sum(weight_matrix[:, k, :, :] * P_k)

        weight_sum = weight_matrix.sum(dim=1, keepdim=True)
        return (cut_value.mean() / weight_sum.squeeze(1).mean())

    # SSIM Loss
    def ssim_loss(img1, img2):
        img1_np = img1.squeeze(1).cpu().detach().numpy()
        img2_np = img2.squeeze(1).cpu().detach().numpy()
        batch_ssim_loss = 0

        for i in range(img1_np.shape[0]):
            win_size = min(7, min(img1_np[i].shape))
            ssim_val = ssim(img1_np[i], img2_np[i], data_range=1, win_size=win_size)
            batch_ssim_loss += (1 - ssim_val)

        return batch_ssim_loss / img1_np.shape[0]



    device ='cpu'

    import torch


    model = WNet(1,64,10)  


    model_path = '/home/23m1522/Wnet/archive/wnet_weights.pth'
    state_dict=torch.load(model_path)

    model.load_state_dict(state_dict)

    # Move model to GPU if available
    model.to( "cpu")

    # Set the model to evaluation mode if using it for inference
    model.eval()
    print("Model loaded successfully.")



    def process_nii_image(nii_file):
    # Load the NIfTI file
        nii_data = nib.load(nii_file)
        
        # Convert to a numpy array
        image_data = nii_data.get_fdata().astype(np.float32)
        
        return image_data




    filtered_image_path = nii_file
    filtered_image = process_nii_image(filtered_image_path)
    label_path = label_input
    label = process_nii_image(label_path)
    plt.imshow(filtered_image,cmap="gray")


    def find_first_minimum_after_peak(hist, start_index):
        """Find the first minimum in the histogram after the specified index."""
        for i in range(start_index + 1, len(hist) - 1):
            if hist[i] < hist[i - 1] and hist[i] < hist[i + 1]:
                return i
        return None  # If no minimum found

    def fill_internal_holes(binary_mask):
        """
    Fill internal holes in the binary liver mask to ensure contiguous segmentation.

        Parameters:
        - binary_mask (np.ndarray): Binary mask (0 for background, 1 for liver).

        Returns:
        - filled_mask (np.ndarray): Binary mask with internal holes filled.
        """
        # Convert mask to 8-bit if not already
        binary_mask = (binary_mask * 255).astype(np.uint8) if binary_mask.max() <= 1 else binary_mask

        # Find contours
        contours, hierarchy = cv2.findContours(binary_mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

        # Fill the holes using the hierarchy information
        for i in range(len(contours)):
            # Check if the contour is an internal hole
            if hierarchy[0][i][3] != -1:  # Parent exists, meaning this is a hole
                cv2.drawContours(binary_mask, contours, i, 255, thickness=cv2.FILLED)

        # Convert back to binary (0 or 1)
        filled_mask = (binary_mask > 0).astype(np.uint8)
        return filled_mask

    def test_wnet_and_segment_liver_single_image(model, filtered_image, num_clusters=10):
        model.eval()
        device = 'cpu'

        # Move HU image to device and add batch and channel dimensions
        filtered_image = torch.tensor(filtered_image, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)

        # Run inference on single image
        with torch.no_grad():
            _, softmax_output, _ = model(filtered_image)
        softmax_output = softmax_output.cpu().numpy().squeeze(0)  # Remove batch dimension

        # Calculate standard deviation for each cluster
        cluster_std_devs = np.zeros(num_clusters)
        for k in range(num_clusters):
            cluster_pixels = softmax_output[k, :, :]
            # Calculate standard deviation for non-zero pixels in the cluster
            cluster_std_devs[k] = np.std(cluster_pixels[cluster_pixels > 0])

        

        # Select clusters with max and min standard deviation
        selected_cluster = np.argmax(cluster_std_devs)
        min_cluster = np.argmin(cluster_std_devs)
        print(f"Cluster with max standard deviation (likely liver): {selected_cluster}")
        # print(f"Cluster with min standard deviation (likely background): {min_cluster}")

        # Display histograms for both selected and minimum standard deviation clusters
        cluster_output = softmax_output[selected_cluster, :, :]
        pixel_values = cluster_output[cluster_output > 0]
        
        min_cluster_output = softmax_output[min_cluster, :, :]
        min_pixel_values = min_cluster_output[min_cluster_output > 0]

        # Plot histograms
        plt.figure(figsize=(10, 6))
        plt.hist(pixel_values, bins=50, color='blue', alpha=0.7)
        plt.title(f'Histogram of Pixel Values for Cluster {selected_cluster} (Max Std Dev)')
        plt.xlabel('Pixel Value')
        plt.ylabel('Frequency')
        plt.grid(axis='y', alpha=0.75)
        plt.show()

        plt.figure(figsize=(10, 6))
        plt.hist(min_pixel_values, bins=50, color='blue', alpha=0.7)
        plt.title(f'Histogram of Pixel Values for Cluster {min_cluster} (Min Std Dev)')
        plt.xlabel('Pixel Value')
        plt.ylabel('Frequency')
        plt.grid(axis='y', alpha=0.75)
        # plt.show()
        hist, bin_edges = np.histogram(pixel_values, bins=256, range=(0, 1))
        peaks, _ = find_peaks(hist)

        # Verify if there are at least two peaks
        if len(peaks) < 2:
            raise ValueError("Not enough peaks found in the histogram. Check the input data or clustering results.")

        # Sort peaks by height to identify larger and smaller peaks
        peak_heights = hist[peaks]
        sorted_peaks = sorted(zip(peaks, peak_heights), key=lambda x: x[1], reverse=True)
        larger_peak_idx, larger_peak_height = sorted_peaks[0]
        smaller_peak_idx, smaller_peak_height = sorted_peaks[1]

        

        
        if smaller_peak_idx < larger_peak_idx:
            # Case 1: Smaller peak appears before the larger peak
            first_minimum_index = find_first_minimum_after_peak(hist, smaller_peak_idx)
            if first_minimum_index is not None:
                first_minimum_value = bin_edges[first_minimum_index]
                print(f"First Minimum after smaller peak (Threshold for <): {first_minimum_value}")
                binary_mask = (cluster_output < first_minimum_value).astype(np.uint8)
            else:
                raise ValueError("No minimum found after the smaller peak.")

        else:
            # Case 2: Smaller peak appears after the larger peak
            # Find the first minimum before the smaller peak
            first_minimum_index = None
            for i in range(smaller_peak_idx - 1, -1, -1):
                if hist[i] < hist[i - 1] and hist[i] < hist[i + 1]:
                    first_minimum_index = i
                    break
            
            if first_minimum_index is not None:
                first_minimum_value = bin_edges[first_minimum_index]
                print(f"First Minimum before smaller peak (Threshold for >): {first_minimum_value}")
                binary_mask = (cluster_output > first_minimum_value).astype(np.uint8)
            else:
                raise ValueError("No minimum found before the smaller peak.")

        
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        ref_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)
        refined_mask = fill_internal_holes(ref_mask)
        return refined_mask, selected_cluster, min_cluster


    mask, selected_cluster, min_cluster = test_wnet_and_segment_liver_single_image(model, filtered_image, num_clusters=10)
    softmax_outputs = []
    # Disable gradient calculations
    with torch.no_grad():
        image = torch.tensor(filtered_image, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
        # Get the model output
        _, softmax_output, _ = model(image)
        softmax_output = softmax_output.cpu().numpy().squeeze(0)
        num_clusters = 10
        # Move output to CPU and convert to numpy
        softmax_outputs.extend(softmax_output)
        
    # Display colormap for selected and minimum standard deviation clusters
    def apply_colormap(cluster_map, title):
        norm_cluster_map = (cluster_map - cluster_map.min()) / (cluster_map.max() - cluster_map.min())
        cmap_ap = cm.jet(norm_cluster_map)  # Applying the colormap
        plt.figure()
        plt.imshow(cmap_ap)
        plt.title(title)
        plt.axis('off')
        plt.show()
        return cmap_ap  # Return the colormap applied image





    def find_centroid(binary_mask):
        # Find contours in the binary mask
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if len(contours) == 0:
            return None  # No contours found

        # Initialize variables to accumulate weighted coordinates and total area
        weighted_x_sum = 0
        weighted_y_sum = 0
        total_area = 0

        # Loop through each contour
        for contour in contours:
            # Calculate area of the contour
            area = cv2.contourArea(contour)
            if area == 0:
                continue  # Skip contours with zero area

            # Calculate moments for the current contour
            moments = cv2.moments(contour)
            
            # Calculate centroid coordinates for this contour
            centroid_x = moments['m10'] / area -20
            centroid_y = moments['m01'] / area -30 
            # Accumulate weighted centroid coordinates and total area
            weighted_x_sum += centroid_x * area
            weighted_y_sum += centroid_y * area
            total_area += area

        if total_area == 0:
            return None  # No valid area found

        # Calculate the weighted average centroid
        avg_centroid_x = weighted_x_sum / total_area 
        avg_centroid_y = weighted_y_sum / total_area 

        return (avg_centroid_x, avg_centroid_y)


    centroid = find_centroid(mask)
    # print(f"centroid of binary maks is {centroid}")


    # Display the mask
    plt.imshow(mask, cmap='gray')
    plt.title('Binary Mask for Liver')
    plt.axis('off')
    # plt.show()




    def apply_colormap_to_binary_mask(binary_mask, colormap='jet'):
        """
        Convert a binary mask to a colored image using a specified colormap.
        
        Parameters:
        - binary_mask (np.ndarray): Binary mask (0 or 1 values).
        - colormap (str): The colormap to apply (default is 'jet').

        Returns:
        - colored_image (np.ndarray): Colored image based on the colormap.
        """
        # Normalize binary mask to [0, 1]
        norm_mask = (binary_mask - binary_mask.min()) / (binary_mask.max() - binary_mask.min())
        
        # Apply colormap to the normalized mask
        cmap_image = cm.get_cmap(colormap)(norm_mask)  # Apply colormap
        cmap_image = (cmap_image[:, :, :3] * 255).astype(np.uint8)  # Remove alpha channel and scale to [0, 255]

        return cmap_image

    def segment_liver_with_sam(binary_mask, coords=None):
        # Load SAM model
        sam_checkpoint = "/home/23m1522/Wnet/sam_vit_h_4b8939.pth"  # Path to SAM checkpoint
        model_type = "vit_h"  # Model type
        device = "cpu"  # Change to "cuda" if using a GPU

        # Initialize the SAM model
        sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
        sam.to(device)

        # Convert the binary mask to a colored image
        colored_mask = apply_colormap_to_binary_mask(binary_mask, colormap='jet')  # Apply 'jet' colormap

        # Ensure colored mask is in RGB format
        if colored_mask.shape[-1] == 4:  # If RGBA, remove alpha channel
            mask_image = colored_mask[..., :3]
        else:
            mask_image = colored_mask  # Already RGB

        # Generate masks for the whole image using SAM
        mask_generator = SamAutomaticMaskGenerator(sam)
        all_masks = mask_generator.generate(mask_image)  # Generate all possible masks

        if coords is not None and all_masks:
            segmented_masks = []

            # Check each coordinate
            for coord in coords:
                coord_x, coord_y = coord

                # Ensure coordinates are within bounds of the mask
                if coord_x < 0 or coord_x >= mask_image.shape[1] or coord_y < 0 or coord_y >= mask_image.shape[0]:
                    print(f"Coordinates out of bounds: {coord}")
                    continue

                found_mask = False
                # Iterate through the generated masks to find one that includes the point
                for mask in all_masks:
                    if mask['segmentation'][coord_y, coord_x]:  # Check if the mask includes the coordinate
                        segmented_masks.append(mask['segmentation'])
                        found_mask = True
                        break

                if not found_mask:
                    print(f"No mask found for coordinates: {coord}")

            # Visualize the segmented masks
            if segmented_masks:
                plt.figure(figsize=(8, 6))
                for i, mask in enumerate(segmented_masks):
                    plt.subplot(1, len(segmented_masks), i + 1)
                    plt.imshow(mask, cmap='gray')
                    plt.title(f"Segmentation at {coords[i]}")
                    plt.axis('off')
                plt.show()
            else:
                print("No valid masks found for the provided coordinates.")
        else:
            print("Coordinates not provided or no masks generated.")

        return segmented_masks


    centroid_int = tuple(map(int, centroid)) if centroid is not None else None
    # if centroid_int:
    #     # Assuming `mask` is the binary mask for liver segmentation
    #     segmented_masks = segment_liver_with_sam(mask, coords=[centroid_int])

                
    def evaluate_segmentation(mask, ground_truth):
        """
        Evaluate the segmentation performance using the predicted mask and ground truth.

        Parameters:
        - mask (np.ndarray): Predicted binary liver mask (1 = liver, 0 = background).
        - ground_truth (np.ndarray): Ground truth binary liver mask (1 = liver, 0 = background).

        Returns:
        - metrics (dict): Dictionary of evaluation metrics.
        """

        # Calculate True Positives, False Positives, True Negatives, and False Negatives
        TP = np.sum((mask == 1) & (ground_truth == 1))  # True Positives: correctly predicted liver pixels
        FP = np.sum((mask == 1) & (ground_truth == 0))  # False Positives: non-liver pixels predicted as liver
        FN = np.sum((mask == 0) & (ground_truth == 1))  # False Negatives: liver pixels missed in prediction
        TN = np.sum((mask == 0) & (ground_truth == 0))  # True Negatives: correctly predicted non-liver pixels

        # Calculate Dice Coefficient
        intersection = np.sum(mask * ground_truth)
        sum_masks = np.sum(mask) + np.sum(ground_truth)
        dice_coefficient = (2. * intersection) / sum_masks if sum_masks != 0 else 1.0
        i_o_u = (dice_coefficient/(2-dice_coefficient))
        # Calculate Accuracy, Precision, Recall, and F1 Score
        accuracy = (TP + TN) / (TP + FP + FN + TN) if (TP + FP + FN + TN) != 0 else 1.0
        precision = TP / (TP + FP) if (TP + FP) != 0 else 0.0
        recall = TP / (TP + FN) if (TP + FN) != 0 else 0.0
        

        # Display results
        print("Segmentation Evaluation Metrics:")
        print("Dice Coefficient:", dice_coefficient)
        print("intersection over union:",i_o_u)
        print("Accuracy:", accuracy)
        print("Precision:", precision)
        print("Recall:", recall)
        print("False Negatives (missed liver pixels):", FN)
        print("False Positives (incorrectly predicted liver pixels):", FP)
        print("True Positives (correctly predicted liver pixels):", TP)
        print("True Negatives (correctly predicted non-liver pixels):", TN)
        plt.imshow(ground_truth,cmap="gray")
        plt.title('Ground Truth')
        plt.axis('off')

        plt.show()
        false_negatives = (mask == 0) & (ground_truth == 1)
        plt.imshow(mask, cmap='gray')
        plt.imshow(false_negatives, cmap='Reds', alpha=0.5)  # Highlight false negatives in red
        plt.title('False Negatives Highlighted')
        plt.axis('off')

        plt.show()



    evaluate_segmentation(segment_liver_with_sam(mask, coords=[centroid_int])[0], label)

    filtered_image_path = nii_file
    filtered_image = process_nii_image(filtered_image_path)
    label_path = label_input
    label = process_nii_image(label_path)
    plt.imshow(filtered_image,cmap="gray")


    pass