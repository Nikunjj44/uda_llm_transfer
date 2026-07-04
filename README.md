# Unsupervised Domain Adaptation on Office-Home (Art → Real World)

This project aims to benchmark two families of methods for handling domain shift in image classification: **output-space self-training** and **CLIP-based knowledge transfer**. The components covered in this repository include self-training approaches (Vanilla ST, CBST, CRST) and Large Multimodal Model (LMM) knowledge transfer methods (Zero-shot CLIP, CLIP-Adapter, Tip-Adapter, Tip-Adapter-F), all evaluated on the Office-Home Art → Real World domain pair.

Unsupervised Domain Adaptation (UDA) addresses the challenge of transferring knowledge learned on a labelled source domain to an unlabelled target domain. Models trained on one visual style almost always disappoint when deployed on data that looks slightly different from what they saw during training. This mismatch is called domain shift, and UDA is the field that tries to fix it without requiring any labels for the target domain. This project explores two contrasting philosophies to bridge this gap: self-training keeps the original source-trained model and lets it learn from its own confident predictions on the target data, whereas CLIP-based transfer discards the original model entirely and adapts a frozen vision-language model with a tiny add-on instead.

## Project Setup

This project operates within the following technical framework:

1. **Task 1 Backbone (Self-Training):** A **ResNet-50 initialized from ImageNet pretrained weights** is used with a 65-way linear classifier head. This choice provides a strong feature extractor while requiring only the classifier head to be adapted for the Office-Home domain.
2. **Task 2 Backbone (LMM Transfer):** A **frozen CLIP ViT-B/16** backbone is used as the vision-language model. CLIP itself is never updated during training; only the small adapter components receive gradients.
3. **Data Constraints:** Self-training methods use no labelled target data (only unlabelled target images for pseudo-labelling). LMM methods use K = 16 support images per class from the source Art dataset (few-shot setup).
4. **Evaluation Protocol:** All methods are evaluated on the full Real World dataset using Top-1 accuracy, macro-F1 (weights every class equally), and weighted-F1 (weights classes by their support).

## Tech Stack Used

![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)
![CLIP](https://img.shields.io/badge/CLIP-Vision%20Language%20Model-blue?style=for-the-badge)
![ResNet](https://img.shields.io/badge/ResNet--50-ImageNet%20Pretrained-orange?style=for-the-badge)
![Domain Adaptation](https://img.shields.io/badge/Domain%20Adaptation-UDA-green?style=for-the-badge)
![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-%23ffffff.svg?style=for-the-badge&logo=Matplotlib&logoColor=black)

## Methodology

1. **Dataset**  
   The **Office-Home** dataset consists of 65 object categories captured in either art forms like paintings (Art dataset with 2,427 images) or in real natural settings (Real World dataset with 4,357 images). The Art domain is used as the labelled source and the Real World domain is used as the unlabelled target. This domain pair captures the ability of a method to bridge the semantic gap between artistic renderings and real-world objects, with challenges including differences in orientation, lighting conditions, and stylistic representation across 65 diverse classes.

2. **Task 1: Output Space Self-Training**  
   Unsupervised Domain Adaptation (UDA) is a transfer learning methodology where a model is trained on a labelled source dataset and then adapted with self-training methods to perform optimally on an unlabelled target dataset. The source-only baseline uses a **ResNet-50 (ImageNet pretrained)** trained for 40 epochs with SGD and a cosine learning rate scheduler. Each self-training method is run for 3 rounds and 3 epochs per round with a portion schedule of 0.20, 0.35, 0.50. This portion schedule implements a curriculum where at each round the pseudo-label coverage expands along with the model's improving reliability. Selecting high values at early rounds would integrate noisy labels before the model is robust enough to absorb them, whereas selecting conservatively in later rounds would not help strengthen the model with useful supervision. Three self-training methods are evaluated: **Vanilla ST** (selects by global confidence), **CBST** (applies the same threshold per predicted class to balance selection), and **CRST** (adds an entropy regularizer on top of CBST to reduce overconfident noisy pseudo-labels).

3. **Task 2: LMM Knowledge Transfer**  
   LMM (Large Multimodal Models) knowledge transfer is the methodology where expertise is transferred from a larger pretrained model into a smaller adapting model for a different or new domain. This task investigates leveraging the pretrained knowledge of a frozen **CLIP ViT-B/16** to improve performance on the Office-Home target domain. K = 16 support images per class (across all 65 classes) are sampled from the source Art dataset, and evaluation is performed on the full Real World dataset. CLIP image features and text features are precomputed with CLIP's preprocessing pipeline. Three adaptation methods are evaluated on top of the frozen CLIP backbone: **Zero-Shot CLIP** (classifies via cosine similarity between image embeddings and class-name text embeddings with no training), **CLIP-Adapter** (a 2-layer MLP bottleneck with reduction ratio of 4, trained with AdamW optimizer and a residual blending weight of 0.1), and **Tip-Adapter / Tip-Adapter-F** (a training-free cache-based method where cache keys are the raw CLIP features of the 16-shot support, with the "-F" variant fine-tuning the cache keys for 20 epochs).

4. **Task 1 Results and Analysis**  
   For Task 1, **CRST performs the best on all metrics with CBST showing near-identical performance**. Both methods improve on the source-only baseline by 1.9% in accuracy and 2.4% in macro-F1, showing that the improvement is concentrated in the minority classes. The larger improvement in macro-F1 values indicates that the gain is due to improvement in accuracy for classes that source-only is failing on. Furthermore, **Vanilla ST regresses on all metrics** because it selects pseudo-labels based on global confidence, which leads to majority class dominance. At round 3 with portion = 0.5, Vanilla ST selects zero pseudo-labels for 13 classes out of 65, whereas CBST covers every class. This severe class imbalance in Vanilla ST's pseudo-labels causes the model to overfit to majority classes like "Knives" or "Computer" and misclassify real-world target images accordingly.

5. **Task 2 Results and Analysis**  
   For Task 2, **all methods improve on Zero-Shot CLIP**, with **Tip-Adapter-F achieving the best overall result** (0.9089 accuracy, 0.8937 macro-F1, 0.9081 weighted-F1). Tip-Adapter is the strongest zero-training method, with the fine-tuned Tip-Adapter-F variant showing a small improvement of 0.0037. Although there is an increase in all metrics compared to Zero-Shot CLIP Adapter, the macro-F1 metric is consistently lower than the other metrics, indicating that the model still struggles with minority classes and highlighting the importance of handling class imbalance in the Office-Home dataset for better generalization. CLIP-Adapter's gain is small but non-trivial, achieved through ablation testing of the residual ratio α. This ablation study also reveals that as the model relies more on the adapter (increasing α), the performance decreases, suggesting that the adapter is advantageous only as a fine-tuner rather than a full replacement of CLIP's features.

6. **Qualitative Observations**  
   For Task 1, qualitative analysis shows Vanilla ST incorrectly predicting majority classes (like "Knives" or "Computer") for real-world target images due to its class-imbalanced pseudo-label selection, whereas CBST and CRST correctly predict the labels. For Task 2, the failure patterns of Zero-Shot CLIP occur on occluded images (e.g., a white cat sitting alongside a mouse) or images with unique visual styles (e.g., green lime soda packaging), suggesting that adapters effectively teach CLIP to focus on the dataset's specific class labels rather than CLIP's broader functional contexts. Additionally, CLIP-Adapter makes more confident predictions compared to Tip-Adapter, but Tip-Adapter achieves slightly higher accuracy — indicating Tip-Adapter is more accurate but CLIP-Adapter is more confident in its predictions. Fine-tuning the cache keys (Tip-Adapter → Tip-Adapter-F) allows the model to transition from slightly identifying the pattern to verifying it with high confidence.

## Key Results

### Task 1: Output-Space Self-Training (Art → Real World)

| Method | Test Accuracy | Macro-F1 | Weighted-F1 |
|--------|--------------|----------|-------------|
| Source-Only | 0.6812 | 0.6363 | 0.6588 |
| Vanilla ST | 0.6702 | 0.6212 | 0.6421 |
| CBST | 0.6991 | 0.6593 | 0.6800 |
| **CRST** | **0.6998** | **0.6603** | **0.6811** |

### Task 2: LMM Knowledge Transfer (Art → Real World)

| Method | Accuracy | Macro-F1 | Weighted-F1 |
|--------|----------|----------|-------------|
| Zero-Shot CLIP | 0.8992 | 0.8818 | 0.8981 |
| CLIP-Adapter | 0.9043 | 0.8911 | 0.9042 |
| Tip-Adapter | 0.9052 | 0.8874 | 0.9035 |
| **Tip-Adapter-F** | **0.9089** | **0.8937** | **0.9081** |

## Future Scope

The complementary failure modes of self-training and CLIP-based transfer point to hybrid approaches as a natural direction for future work. On this domain pair, CLIP-based methods significantly outperform self-training (0.90 vs 0.70 accuracy), suggesting that leveraging the pretraining coverage of large vision-language models is highly effective when the target domain resembles data that CLIP has likely seen during pretraining. Further work could explore combining the confidence regularization ideas from CRST with the adapter-based fine-tuning of Tip-Adapter-F to address minority class handling more effectively.

## Group Contribution

This project was completed as a group effort. The Art → Real World (Office-Home) portion documented in this README represents my individual contribution:

* **Pentamsetty Sai Harshita** (G2503340A)
* **Tenbarai Sriram Sri Hari** (G2503355F)
* **Birul Gulam Mustafa** (G2504112L)
* **Madan Nikunj** (G2504045E) — *Art → Real World (Office-Home)*

## Academic Context
Completed during my Graduate studies at **Nanyang Technological University (NTU), Singapore**.

* **Course:** AI6126 - Advanced Computer Vision
* **Project:** Project 2 - Output Space UDA and LMM Transfer
* **Supervisor:** Prof. Lu Shijian
* **Semester:** AY 2025/2026, Semester 1

## Usage
**Note for current/future NTU students:** While this repository is public, please ensure you adhere to NTU's Academic Integrity Policy. This is intended as a reference for my personal portfolio; using this code for your own graded assignments is strictly prohibited by the University.
