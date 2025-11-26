# Lung Segmentation using Residual U-Net

This project implements a deep learning pipeline for Lung Segmentation from CT images. It builds upon the standard U-Net architecture by incorporating Residual Blocks, advanced data augmentation, and a combined Dice-BCE loss function to achieve higher Intersection over Union (IoU) scores.

## Key Improvements

1.  **Residual U-Net Architecture**:
    -   Replaced standard convolutional blocks with **Residual Blocks**.
    -   Residual connections help in training deeper networks by mitigating the vanishing gradient problem and allowing the model to learn identity mappings.
    -   This leads to better feature extraction and faster convergence.

2.  **Advanced Loss Function**:
    -   Used **Dice Loss + Binary Cross Entropy (BCE) Loss**.
    -   BCE works well for pixel-level classification but can struggle with class imbalance (background vs. lung).
    -   Dice Loss directly optimizes the IoU metric, ensuring the model focuses on the overlap between predicted and ground truth masks.

3.  **Robust Data Augmentation**:
    -   Utilized the `albumentations` library for fast and effective augmentations.
    -   Included **Rotation, Horizontal/Vertical Flips, and Normalization**.
    -   This improves the model's generalization capability and prevents overfitting.

4.  **IoU Metric Tracking**:
    -   Implemented a custom Intersection over Union (IoU) metric to monitor segmentation performance during training and validation.

## Project Structure

-   `lung_segmentation.py`: The main script containing the configuration, dataset, model, training loop, and evaluation logic.
-   `requirements.txt`: List of Python dependencies.
-   `Lung_Segmentation.ipynb`: Jupyter Notebook with training visualization.

## Installation

1.  Clone the repository.
2.  Install the dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

1.  **Prepare Data**:
    -   Ensure your lung CT images and masks are organized in the following structure:
        ```
        Train/
        ├── Images/
        ├── Masks/
        Test/
        ├── Images/
        └── Masks/
        ```
    -   Update the paths in the `Config` class in `lung_segmentation.py` if your data is located elsewhere.

2.  **Run Training**:
    ```bash
    python lung_segmentation.py
    ```

3.  **Output**:
    -   The script will print the Loss and IoU for each epoch.
    -   The best model (highest validation IoU) will be saved as `best_resunet_model.pth`.
    -   (Optional) Uncomment the `save_predictions_as_imgs` call in the main loop to visualize predictions.

## Model Architecture Details

The **ResUNet** consists of:
-   **Encoder**: 4 downsampling stages using Residual Blocks.
-   **Bottleneck**: A Residual Block connecting encoder and decoder.
-   **Decoder**: 4 upsampling stages using Transpose Convolutions and Residual Blocks.
-   **Skip Connections**: Standard U-Net skip connections concatenated with upsampled features to preserve spatial information.

## Results

By combining residual learning with a segmentation-specific loss function, this model aims to surpass the baseline U-Net performance, particularly in boundary delineation and handling variable lung shapes.
