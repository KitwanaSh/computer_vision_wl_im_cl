# Wildlife Image Classification Competition
![Image example]("./data/image_screen.png")

As a computer vision engineer, I tackled this competition to classify wildlife images into eight categories: `antelope_duiker`, `bird`, `blank`, `civet_genet`, `hog`, `leopard`, `monkey_prosimian`, and `rodent`. The goal was to develop a robust model using a limited training set, validate its performance, and generate predictions for the test set submission. Below, I document the process, challenges, and solutions from start to finish.

## Dataset Preparation

The journey began with the training data (`train_labels.csv` and `train_features/`), containing 16,000 images with labels across the eight species. Each row had an `id`, `filepath`, and one-hot-encoded species columns. The validation and test sets (`test_features.csv`) followed a similar structure, with `id` as the index, `filepath`, and an additional `site` column for test data, but no labels for the latter.

To handle this, I built a custom PyTorch `Dataset` class, `ConserVisionDataset`, to:
- Load images from `filepath` using PIL.
- Apply transformations: resize to 224x224, convert to tensor, and normalize with ImageNet stats (`mean=[0.485, 0.456, 0.406]`, `std=[0.229, 0.224, 0.225]`).
- Convert one-hot labels to class indices (`label_idx`) via `idxmax`.

For the test set, I adapted this into `TestConserVisionDataset`, skipping label processing since none were provided.

**Challenge**: The initial 16,000-image dataset was too large for my local CPU, causing training to take "forever." I reduced the training set to 1000 images and validation to 800 via random sampling, balancing speed and learning capacity.

## Model Selection and Training

I chose a pre-trained ResNet18 from `torchvision`, leveraging its ImageNet features for transfer learning. The final fully connected layer was modified from 1000 classes to 8 (`model.fc = nn.Linear(model.fc.in_features, 8)`). This lightweight architecture suited the small dataset and local compute constraints.

Training ran on my local CPU (Windows), with:
- **Optimizer**: Adam, learning rate 0.001.
- **Loss**: CrossEntropyLoss.
- **Batch Size**: 64.
- **Epochs**: 5 (initially planned 15, scaled back due to time).
- **DataLoader**: `num_workers=0` (Windows compatibility), `pin_memory=False`.

A custom `train_model` function tracked loss and accuracy per epoch, using a `ProgressCallback` for real-time monitoring. After 5 epochs (~10.5 minutes), the training accuracy reached 74.2% and loss dropped to 0.7283, showing solid learning on the reduced set.

**Challenge**: Early attempts on the full 16,000 images were infeasible locally. Switching to Google Colab’s GPU was considered, but I stuck with local training after reducing the dataset, achieving a manageable runtime.

## Validation and Evaluation

With 800 validation images (same structure as training), I loaded the saved model (`resnet18_5epochs.pth`) and ran predictions. The validation accuracy was 61.75% with a loss of 1.2787—lower than training, indicating mild overfitting due to the small training size, but still promising.

To understand performance, I:
- **Plotted Loss and Accuracy**: Training loss fell from 1.6790 to 0.7283, and accuracy rose from 38.5% to 74.2% over 5 epochs. Validation loss per batch was stable around 1.2787.
- **Distribution**: Compared predicted vs. true class counts to check for bias.
- **Confusion Matrix**: Highlighted where misclassifications occurred (e.g., confusing similar species like `bird` and `monkey_prosimian`).

**Visualization Screenshot**  
Here’s the loss and accuracy over time from training (insert your screenshot):  
![Loss and Accuracy Over Time]("./data/loss_accuracy.png")

**Challenge**: The validation set initially lacked labels, requiring a fix by ensuring it mirrored the training structure. This let me compute true metrics and refine the process.

## Submission Preparation

For the test set (`test_features.csv`), I predicted probabilities for all 8 classes using the trained model. The submission format required `id` as the index and one column per class with softmax probabilities. I:
- Loaded `test_features` (index: `id`, columns: `filepath`, `site`).
- Used `TestConserVisionDataset` and a `DataLoader` (batch size 64).
- Ran inference, collecting probabilities via `F.softmax`.
- Created `submission_df` with `id` as the named index and columns matching the class names.


**Challenge**: Ensuring the index was named `'id'` in the CSV required explicitly setting `submission_df.index.name = "id"`.

## Key Takeaways

- **Small Data Success**: 74.2% training and 61.75% validation accuracy with just 1000 training images show ResNet18’s transfer learning power.
- **Local Constraints**: Reducing the dataset made CPU training viable, though more data or GPU use could boost generalization.
- **Next Steps**: Scaling to the full 16,000 images on Colab or adding regularization (e.g., dropout) could close the training-validation gap.

This submission reflects a practical, iterative approach to wildlife classification—balancing compute limits with model performance. Ready to see how it scores!

Reach out to me on my [LinkedIn here]("https://www.linkedin.com/in/kiwanasheb/")