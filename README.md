# C-NMC Leukemia Detection

**Distributed Deep Learning for Early-Stage Leukemia Detection on AWS**

A production-scale machine learning system for classifying leukemic B-lymphoblast cells from normal B-lymphoid precursors using microscopic images. Implements distributed deep learning on Amazon Web Services (AWS) using Apache Spark, SageMaker, and EMR.

## 🎯 Overview

This project addresses a critical healthcare challenge: early detection of leukemia through automated analysis of microscopic cell images. Traditional detection requires expensive medical expertise and often catches cancer only in advanced stages. This automated system provides:

- **Early Detection** - Identifies malignant cells before they proliferate
- **Cost-Effective** - Reduces dependency on scarce medical experts
- **Scalable** - Distributed computing handles large-scale medical imaging datasets
- **Accurate** - Deep learning achieves high classification accuracy

## 🏥 Medical Context

**Problem**: Morphologically, malignant and normal lymphoblast cells appear very similar under microscopy, making manual classification difficult and time-consuming.

**Solution**: Deep learning image classification trained on thousands of labeled cell images to distinguish cancerous from healthy cells with high accuracy.

**Impact**: Enables earlier diagnosis, better treatment outcomes, and improved patient survival rates.

## 📊 Dataset

- **Source**: CodaLab Competition - C-NMC Challenge
- **Size**: 118 patients, 25,794+ cell images
- **Image Size**: ~300x300 pixels (preprocessed, normalized, segmented)
- **Classes**:
  - **ALL** (Acute Lymphoblastic Leukemia) - Malignant cells
  - **HEM** (Hematopoietic) - Normal/healthy cells
- **Split**:
  - Training: 73 patients (47 cancer, 26 healthy)
  - Testing: 45 patients (validation sets)

### Naming Convention:
```
UID_P_N_C_diagnosis
  P = Patient ID
  N = Image number
  C = Cell count
  diagnosis = 'all' (cancer) or 'hem' (healthy)
```

## 🏗️ Architecture & Implementation

### Distributed Computing Stack:

1. **Amazon S3** - Scalable data storage for 800MB+ image dataset
2. **Apache Spark** - Distributed data processing framework
3. **Amazon SageMaker** - Managed ML platform with Spark integration
4. **Amazon EMR** - Elastic MapReduce cluster for distributed training

### Two Parallel Approaches:

#### Approach 1: SageMaker with Transfer Learning
- Pre-trained deep learning model (VGG/ResNet/InceptionV3)
- Fine-tuned on medical imaging data
- Optimized using Spark for data preprocessing
- Scalable training pipeline

#### Approach 2: EMR + TensorFlow on Spark
- Custom CNN architecture built with TensorFlow
- Distributed training across EMR cluster nodes
- Efficient batch processing with Spark DataFrames
- Real-time model evaluation

## 🔬 Technologies Used

### Cloud Infrastructure:
- **AWS S3** - Data lake storage
- **AWS SageMaker** - ML training & deployment
- **AWS EMR** - Distributed compute cluster

### Frameworks & Libraries:
- **Apache Spark (PySpark)** - Distributed computing
- **TensorFlow/Keras** - Deep learning models
- **Transfer Learning** - Pre-trained CNN architectures
- **Python** - Primary programming language
- **NumPy, Pandas** - Data manipulation
- **OpenCV** - Image preprocessing

## 📈 Results & Evaluation

**Metric**: F1 Score (balanced precision and recall for medical applications)

[Add your model performance results here]

- Training Accuracy: [X%]
- Validation Accuracy: [X%]
- F1 Score: [X]
- Sensitivity (Recall): [X%] - Critical for cancer detection
- Specificity: [X%] - Minimize false positives

## 🚀 Setup & Installation

### Prerequisites:
- AWS Account with S3, SageMaker, and EMR access
- Python 3.7+
- Apache Spark 3.x

### Data Preparation:
```bash
# Data is already preprocessed and segmented
# Training data: C-NMC_training_data/
# Test data: C-NMC_test_prelim_phase_data/ & C-NMC_test_final_phase_data/
```

### Running on SageMaker:
1. Upload data to S3 bucket
2. Create SageMaker notebook instance with Spark
3. Run transfer learning pipeline
4. Deploy trained model

### Running on EMR:
1. Provision EMR cluster with TensorFlow
2. Configure Spark context
3. Execute distributed training script
4. Evaluate on test set

## 📁 Project Structure

```
C-NMC_Leukemia/
├── C-NMC_training_data/       # Training images (all/hem)
├── C-NMC_test_prelim_phase_data/  # Preliminary test set
├── C-NMC_test_final_phase_data/   # Final test set
├── preprocess_data/           # Preprocessed datasets
│   ├── training_data/
│   │   ├── all/              # Cancer cell images
│   │   └── hem/              # Healthy cell images
│   └── validation_data/
├── CNMC_readme.pdf            # Dataset documentation
├── Report.pdf                 # Full project report
└── README.md                  # This file
```

## 📚 Key Learnings

- **Distributed Deep Learning** - Training CNNs on Spark clusters
- **AWS Cloud Architecture** - S3, SageMaker, EMR integration
- **Transfer Learning** - Adapting pre-trained models for medical imaging
- **Medical AI** - Handling class imbalance and high-stakes predictions
- **Big Data Processing** - Managing 800MB+ image datasets efficiently
- **Model Optimization** - Balancing accuracy, speed, and cost on cloud infrastructure

## 🔮 Future Enhancements

- Deploy as real-time inference API
- Implement ensemble methods (multiple models)
- Add explainability (GradCAM visualization)
- Expand to multi-class cancer classification
- Mobile app for point-of-care diagnosis
- Integration with hospital PACS systems

## 📖 References

- CodaLab C-NMC Challenge Dataset
- [Transfer Learning Paper](relevant link)
- Apache Spark MLlib Documentation
- AWS SageMaker Best Practices
- TensorFlow on Spark

## 📄 Documentation

- Full project report: `Report.pdf`
- Dataset documentation: `CNMC_readme.pdf`

## 🏆 Impact

This project demonstrates how distributed cloud computing and deep learning can democratize access to advanced medical diagnostics, potentially saving lives through earlier cancer detection.

## Contact

- GitHub: [@dyadav4](https://github.com/dyadav4)
