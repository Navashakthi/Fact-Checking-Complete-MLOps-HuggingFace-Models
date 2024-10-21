# Fact-Checking-Complete-MLOps-HuggingFace-Models
Task of verifying the veracity of claims using hugging face models and kuberbetes deployment

When setting weights for an NLP fact-checking model, prioritize metrics that reflect its accuracy in classifying claims as true, false, or unproven, while considering the impact of misclassifications. Suggested weights are:

- **Accuracy (20%)**: Measures overall correctness but is less informative for imbalanced datasets.
- **Loss (20%)**: Indicates model learning; minimizing it is crucial for training but less critical for final evaluation.
- **F1 Score (30%)**: Balances precision and recall, essential for addressing imbalanced classes in fact-checking.
- **Micro F1 (15%)**: Assesses overall model performance across all classes, aiding generalization.
- **Macro F1 (15%)**: Treats all classes equally, ensuring good performance on all claim types (true, false, unproven).
