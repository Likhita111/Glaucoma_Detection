# Glaucoma Eye Disease Detection using Machine Learning (GlaucoVision)

## 🧠 About the Project

**GlaucoVision** is a full-stack machine learning-based web application designed to assist in the early detection of **glaucoma**, a serious eye condition that can lead to blindness if left untreated. By leveraging **transfer learning** with pre-trained CNN architectures like **ResNet50**, **InceptionV3**, **EfficientNetB7**, and **VGG16**, this project aims to provide accurate and efficient glaucoma screening from fundus images.

The core technologies powering this system are:

* **Frontend**: React.js
* **Backend**: Flask
* **ML Models**: Python (TensorFlow, Keras)

## ⚙️ Project Setup

This project has two parts:

1. **Model Training** – where the glaucoma detection models are built and trained.
2. **Web Application** – where the trained models are integrated into a full-stack web app.

### Prerequisites

* [Git](https://git-scm.com/)
* [Python](https://www.python.org/)
* [Node.js](https://nodejs.org/en)
* [Miniconda](https://docs.anaconda.com/free/miniconda/) or [Anaconda](https://www.anaconda.com/)

Clone the repository:

```bash
git clone https://github.com/<your-username>/glaucoma-detection-ml.git
cd glaucoma-detection-ml
```

---

## 🧪 Model Training

Model training notebooks and datasets are located in the `model/` directory. The training uses **transfer learning** with:

* `ResNet50`
* `InceptionV3`
* `EfficientNetB7`
* `VGG16`

### Using Google Colab

Download and run the notebook:

* [glaucoma-detection-model.ipynb](model/glaucoma-detection-model.ipynb)

> ⚠️ Avoid using GPU/TPU if training from scratch, as it may consume your quota.

### Using Local Jupyter Notebook

```bash
# Setup Conda environment
conda env create -f model/environment.yml
conda activate glaucoma-detection
```

> 💡 Use VSCode or Jupyter Lab and select the correct Python interpreter.

---

## 🌐 Web Application

### Frontend (React)

```bash
cd client
npm install
npm run dev
```

### Backend (Flask)

```bash
cd server
conda env create -f server/environment.yml
conda activate glaucoma-detection-server
python server.py
```

Alternatively, for `pip` users:

```bash
pip install -r server/requirements.txt
```

**Main Backend Dependencies:**

* Flask
* Flask-CORS
* TensorFlow
* Scikit-learn
* Scikit-image
* Numpy

---

## 🔍 Algorithms Used

This project explores the performance of multiple deep learning models using **transfer learning**:

| Model              | Description                                                               |
| ------------------ | ------------------------------------------------------------------------- |
| **ResNet50**       | Deep CNN with residual blocks to mitigate vanishing gradients.            |
| **InceptionV3**    | Multi-scale convolutional architecture to capture varied features.        |
| **EfficientNetB7** | High accuracy and performance-optimized CNN with compound scaling.        |
| **VGG16**          | Deep CNN with uniform architecture; simple yet effective for image tasks. |

---

## 🙌 Contributing

We welcome contributors! Follow these steps:

1. Fork the repository.
2. Setup the project and run it locally.
3. Raise an issue before submitting a Pull Request (PR).
4. Discuss your changes in the [Discussion Section](https://github.com/<your-repo>/discussions).

> 🚫 **Note**: PRs without linked issues will be closed.

---

## 📬 Contact

If you face any issues or want to contribute ideas, feel free to open a [GitHub Issue](https://github.com/<your-repo>/issues).

