# Symmetry Lens

Our method combines ideas from multiple fields involving group theory, information theory and signal processing. At the core, our motivation is to learn symmetries of datasets which also leads to a sensible notion of locality. This approach provides us three major benefits:

- Ability to learn symmetries at higher dimensional datasets 
- Ability to learn symmetries even when they act locally
- Providing natural coordinates to represent data. Such that under some circumstances, our method shows the true nature of data.

Currently, we have tested the method on 33 dimensional datasets and one parameter family of symmetry transformations. However, results are encouraging in terms of stability of learning symmetries which motivates us towards pushing the boundaries of this new symmetry and representation learning paradigm. 

For more details, please refer to our paper:

[SymmetryLens: A new candidate paradigm for unsupervised symmetry learning via locality and equivariance](https://arxiv.org/abs/2410.05232)

## Table of Contents

- [Symmetry Lens:](#symmetry-lens)
  - [Table of Contents](#table-of-contents)
  - [Installation](#installation)
    - [Prerequisites](#prerequisites)
    - [Using Conda (Without GPU)](#using-conda-without-gpu)
    - [Using Conda (With GPU)](#using-conda-with-gpu)
    - [Direct installation (Without GPU)](#direct-installation-without-gpu)
    - [Direct installation (With GPU)](#direct-installation-with-gpu)
  - [Training Models](#training-models)
  - [Testing Models](#testing-models)
  - [Possible Use Cases](#possible-use-cases)
    - [Adapter to CNNs for symmetry conversion](#adapter-to-cnns-for-symmetry-conversion)
    - [Inverse problems in wave propagation](#inverse-problems-in-wave-propagation)
    - [Consistent interpretation of raw data](#consistent-interpretation-of-raw-data)
  - [License](#license)
  - [Contact](#contact)

---

## Installation

### Prerequisites

- **Python Version:** Ensure you are using **Python 3.10**.
- **NVIDIA GPU Drivers (If using GPU):** Required for GPU support.
- **CUDA and cuDNN Libraries (If using GPU):** Compatible versions for TensorFlow 2.14.0.

### Using Conda (Without GPU)
- **Create and activate conda environment**
   ```bash
   conda create -n <environment_name> python=3.10
   conda activate <environment_name>
   ```
- **Install package SymmetryLens**
   ```bash
   git clone git@github.com:onurefe/SymmetryLens.git
   cd SymmetryLens
   python setup.py install
   ```

### Using Conda (With GPU)
- **Create and activate conda environment**
   ```bash
   conda create -n <environment_name> python=3.10
   conda activate <environment_name>
   ```

- **Install tensorflow and cuda/cudnn libraries**
   ```bash
   pip install tensorflow[and-cuda]==2.14
   ```

- **Install package SymmetryLens**
   ```bash
   git clone git@github.com:onurefe/SymmetryLens.git
   cd SymmetryLens
   python setup.py install
   ```

### Direct installation (Without GPU)
- **Install package SymmetryLens**
   ```bash
   git clone git@github.com:onurefe/SymmetryLens.git
   cd SymmetryLens
   python setup.py install
   ```

### Direct installation (With GPU)
- **Install tensorflow and cuda/cudnn libraries**
   ```bash
   pip install tensorflow[and-cuda]==2.14
   ```

- **Install package SymmetryLens**
   ```bash
   git clone git@github.com:onurefe/SymmetryLens.git
   cd SymmetryLens
   python setup.py install
   ```

## Training Models
Utilize the provided **"notebooks/train.ipynb"** notebook to train your models.

- ***Toy Models***:
Use the synthetic dataset generator included in the notebook for quick experiments and testing. 

- ***Custom Applications***:
  1. Prepare Your Dataset: Create a dataset with the shape [BATCH_SIZE, TIMESTEPS, 1]. Note: Currently, multi-channel data isn't supported directly. As a workaround, you can use multiple layers in parallel.

  2. Train the Model: Call the train function with appropriate parameters. Input your raw data without any preprocessing steps.
   
## Testing Models
Use the **"train.ipynb"** notebook as template to evaluate your models.

- ***Verify Symmetry Learning***:
Check if the model has successfully learned the intended symmetries.

- ***Interpretable Data Representation***:
Leverage the learned symmetry to represent and analyze data samples in an interpretable manner.

## Possible Use Cases

### Adapter to CNNs for symmetry conversion
After training the Group Correlation Layer, the action of the learned symmetry manifests as translation in the canonical representation. This feature allows the use of Convolutional Neural Networks (CNNs) on top of it, resembling a Group Equivariant Network. This setup enhances the model's ability to recognize patterns regardless of their position or orientation.

### Inverse problems in wave propagation
Apply the package to solve inverse problems related to wave propagation in dispersive media. By leveraging symmetry-aware models, you can achieve more accurate reconstructions and interpretations of wave behaviors in complex environments.

### Consistent interpretation of raw data
Use the learned symmetries to consistently interpret raw data across different samples or conditions. This consistency aids in creating robust models that generalize well to unseen data by focusing on invariant features.

### Flow learning from many-body systems
Implement flow-based learning techniques to model and understand the dynamics of many-body or ensemble dynamical systems. The symmetry-aware approach ensures that the model captures essential interactions and behaviors inherent in such complex systems.

## Citation
If you find this work useful in your research, please consider citing:

@misc{efe2024symmetrylensnewcandidateparadigm,
      title={SymmetryLens: A new candidate paradigm for unsupervised symmetry learning via locality and equivariance}, 
      author={Onur Efe and Arkadas Ozakin},
      year={2024},
      eprint={2410.05232},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2410.05232}, 
}

## Contact
For any questions, issues, or feature requests, please open an issue on the GitHub repository or contact onur.efe44@gmail.com.
