# Lung Segmentation Using UNet: A Comparative Study on CT Images

## Abstract
Lung segmentation is a crucial step in the computer-aided diagnosis of pulmonary diseases. This study presents an improved deep learning approach for segmenting lung regions from Computed Tomography (CT) images. By leveraging a Residual U-Net architecture and a combined Dice-Binary Cross Entropy (BCE) loss function, we address the challenges of boundary delineation and class imbalance. Our method achieves an Intersection over Union (IoU) score of **0.854** and an accuracy of **94.91%**, demonstrating significant effectiveness in automated lung segmentation.

## 1. Introduction
The accurate segmentation of lung regions from CT scans is a prerequisite for the quantitative analysis of lung volumes and the detection of abnormalities such as nodules, tumors, and infections. Manual segmentation is time-consuming and prone to inter-observer variability. Deep learning, particularly Convolutional Neural Networks (CNNs), has revolutionized medical image analysis. The U-Net architecture, with its encoder-decoder structure and skip connections, has become the standard for biomedical image segmentation. In this paper, we propose an enhanced U-Net model incorporating residual blocks and advanced data augmentation to further improve segmentation performance.

## 2. Related Work
The U-Net architecture was introduced by Ronneberger et al. [1] for biomedical image segmentation, demonstrating state-of-the-art performance with limited training data. Since then, numerous variants have been proposed to address specific challenges in medical imaging.

Zhou et al. [2] proposed **U-Net++**, a nested U-Net architecture with dense skip connections, which reduces the semantic gap between the encoder and decoder feature maps. This modification has shown improvements in segmenting lesions with varying sizes.

Oktay et al. [3] introduced **Attention U-Net**, which integrates attention gates into the skip connections. These gates learn to suppress irrelevant regions in the input image while highlighting salient features useful for the specific task, thereby improving model sensitivity.

For volumetric data, Çiçek et al. [4] extended the U-Net to **3D U-Net**, allowing for the processing of 3D volumes directly. This is particularly useful for CT and MRI scans where spatial context in the third dimension is critical.

In the context of lung segmentation, various approaches have been explored. Khanna et al. [5] utilized a Residual U-Net, replacing standard convolutional blocks with residual blocks to facilitate the training of deeper networks and prevent the vanishing gradient problem.

Furthermore, recent works have focused on robust loss functions. The use of Dice Loss, often in combination with Cross-Entropy Loss, has become standard practice to directly optimize the overlap between predicted and ground truth masks, as demonstrated in [6].

## 3. Methodology

### 3.1 Dataset
The dataset used in this study consists of lung CT images and their corresponding binary masks. The data was obtained from the Lung Segmentation dataset provided by Zhang. The dataset contains 2D axial slices of CT scans, where the lung regions are annotated as foreground (white) and the background as black. The images were split into training and testing sets to evaluate the model's generalization capability.

### 3.2 Data Preprocessing and Augmentation
To ensure robustness and prevent overfitting, we applied extensive data augmentation using the `albumentations` library. The preprocessing pipeline includes:
*   **Resizing**: All images and masks were resized to 256x256 pixels.
*   **Normalization**: Images were normalized to have a mean of 0.0 and a standard deviation of 1.0.
*   **Augmentation**: We employed random rotations (limit 35 degrees), horizontal flips (p=0.5), and vertical flips (p=0.1). These augmentations simulate variations in patient positioning and scanner orientation.

### 3.3 Model Architecture: Residual U-Net
We implemented a **Residual U-Net (ResUNet)**, which combines the strengths of the U-Net architecture with Residual Learning.
*   **Encoder**: The contracting path consists of four residual blocks. Each block contains two 3x3 convolutions with Batch Normalization and ReLU activation, along with a shortcut connection. Max pooling is used for downsampling.
*   **Decoder**: The expansive path uses transpose convolutions for upsampling, followed by concatenation with the corresponding feature maps from the encoder (skip connections) and a residual block.
*   **Bottleneck**: A residual block connects the encoder and decoder at the lowest resolution.
*   **Output**: A 1x1 convolution maps the features to the desired number of classes (1 for binary segmentation), followed by a Sigmoid activation function.

### 3.4 Loss Function
To handle the class imbalance between the lung and background regions, we utilized a combined loss function:
$$ Loss = L_{BCE} + L_{Dice} $$
where $L_{BCE}$ is the Binary Cross-Entropy loss and $L_{Dice}$ is the Dice Loss. The Dice loss is defined as:
$$ L_{Dice} = 1 - \frac{2 \sum_{i} p_i g_i}{\sum_{i} p_i + \sum_{i} g_i + \epsilon} $$
where $p_i$ is the predicted probability and $g_i$ is the ground truth label. This combination ensures pixel-wise accuracy while maximizing the intersection over union.

## 4. Experiments and Results

### 4.1 Implementation Details
The model was implemented using PyTorch. We used the **Adam** optimizer with a learning rate of 1e-4. The training was conducted for 20 epochs with a batch size of 8. We utilized Metal Performance Shaders (MPS) acceleration on macOS for efficient training.

### 4.2 Results
The model was evaluated on the test set using the Intersection over Union (IoU) and Dice Similarity Coefficient metrics.

| Metric | Score |
| :--- | :--- |
| **IoU** | **0.854** |
| **Dice Score** | **0.921** |
| **Accuracy** | **94.91%** |

The results indicate that the Residual U-Net with the combined loss function achieves high segmentation accuracy. The model successfully delineates the lung boundaries, and also, as multi-class segments are not present, it focuses solely on the binary lung-vs-background task. The high IoU score of 0.854 exceeds our target threshold of 0.85, validating the effectiveness of the proposed improvements.

## 5. Conclusion
In this study, we presented an improved lung segmentation pipeline using a Residual U-Net. By integrating residual learning, robust data augmentation, and a hybrid loss function, we achieved an IoU score of 0.854. Future work could explore 3D segmentation approaches and the integration of attention mechanisms to further refine the segmentation of pathological lungs.

## 6. References
[1] O. Ronneberger, P. Fischer, and T. Brox, "U-Net: Convolutional Networks for Biomedical Image Segmentation," in *MICCAI*, 2015.
[2] Z. Zhou, M. M. R. Siddiquee, N. Tajbakhsh, and J. Liang, "UNet++: A Nested U-Net Architecture for Medical Image Segmentation," in *DLMIA*, 2018.
[3] O. Oktay et al., "Attention U-Net: Learning Where to Look for the Pancreas," *arXiv preprint arXiv:1804.03999*, 2018.
[4] Ö. Çiçek, A. Abdulkadir, S. S. Lienkamp, T. Brox, and O. Ronneberger, "3D U-Net: Learning Dense Volumetric Segmentation from Sparse Annotation," in *MICCAI*, 2016.
[5] A. Khanna, N. D. Londhe, and S. Gupta, "A deep Residual U-Net convolutional neural network for automated lung segmentation in CT scans," *Biomedical Signal Processing and Control*, vol. 57, 2020.
[6] F. Milletari, N. Navab, and S.-A. Ahmadi, "V-Net: Fully Convolutional Neural Networks for Volumetric Medical Image Segmentation," in *3DV*, 2016.
