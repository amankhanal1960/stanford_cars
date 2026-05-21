# Fine-Grained Vehicle Classification on Stanford Cars Dataset

A highly optimized deep learning pipeline built using **Keras 3** and **TensorFlow** to tackle fine-grained image classification on the **Stanford Cars Dataset** (196 classes).

This project explores the complexities of fine-grained classification under strict data density constraints (~41 images per class) and demonstrates structural regularization techniques to scale up image resolution to $300 \times 300$ using an **EfficientNetB3** architecture on consumer-grade laptop hardware.

## Performance Highlights

- **Validation Accuracy:** Peak at **~79%** for fine-grained sub-categories.
- **Pipeline Efficiency:** Optimized data loading via offline preprocessing, reducing training time by over 1 minute per epoch.
- **Overfitting Defense:** Reduced the train-to-validation accuracy gap from an aggressive 25% down to a stable margin using advanced structural regularization.

---

## Key Pipeline Architecture

### 1. Offline Bounding-Box Cropping

Fine-grained vehicle recognition requires the model to focus entirely on micro-details (grilles, headlights, badge trims) rather than background clutter (dealerships, trees, roads). A standalone preprocessing script extracts bounding box coordinates from the original `.mat` annotations, crops the cars out natively, and scales them directly to $300 \times 300$ pixels. This saves immense CPU overhead during training loops.

### 2. Regularized Classification Head

To prevent the model from hard-memorizing images given the low sample count, the custom head enforces strict architectural barriers:

- **Batch Normalization:** Stabilizes feature scaling out of the frozen backbone.
- **High Dropout (0.5):** Randomly breaks neuron dependencies to ensure robust feature extraction.
- **L2 Regularization (Weight Decay):** Penalizes extreme weight configurations in the final dense classification layer.

### 3. Strategic Fine-Tuning

Instead of unfreezing the entire backbone—which triggers catastrophic forgetting on small datasets—this model restricts fine-tuning exclusively to the top **30 layers** of the EfficientNetB3 structure during Phase 2.

The pipeline is validated inside a localized Miniconda environment running under Python 3.10 with native CUDA/cuDNN GPU bindings.

# Clone the repository

git clone

# Create and activate environment

conda create -n tf_gpu python=3.10
conda activate tf_gpu

# Install dependencies

pip install -r requirements.txt

OR you can directly run the notebook .ipynb at once as it covers everything.
