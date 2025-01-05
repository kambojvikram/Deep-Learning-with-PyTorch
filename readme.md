
# PyTorch Learning Repository

Welcome to the **PyTorch Learning Repository**! This repository serves as a comprehensive guide to PyTorch concepts, operations, and techniques, helping you master this powerful deep learning framework.

---

## 🔥 **What is PyTorch?**
[PyTorch](https://pytorch.org/) is an open-source machine learning framework that provides dynamic computation graphs, making it intuitive and flexible for building and training deep learning models. Its seamless integration with Python makes it a favorite among researchers and developers.

---

## 📚 **Contents**

1. [Getting Started](#getting-started)
2. [Tensor Basics](#tensor-basics)
3. [Tensor Manipulations](#tensor-manipulations)
4. [Matrix Operations](#matrix-operations)
5. [Tensor Indexing](#tensor-indexing)
6. [Broadcasting and Expanding](#broadcasting-and-expanding)
7. [Tensor Reduction](#tensor-reduction)
8. [Linear Algebra](#linear-algebra)
9. [Comparison Operations](#comparison-operations)
10. [Additional Resources](#additional-resources)

---

## 🚀 **Getting Started**

1. **Install PyTorch**:
   Follow the [official installation guide](https://pytorch.org/get-started/locally/).

   ```bash
   # Example for Linux with CUDA
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   ```

2. **Import PyTorch**:
   ```python
   import torch
   ```

---

## 🧮 **Tensor Basics**

- **Creating Tensors**:
  ```python
  x = torch.tensor([1, 2, 3])  # 1D tensor
  y = torch.zeros(3, 3)        # 2D tensor
  z = torch.rand(3, 4)         # Random tensor
  ```

- **Inspecting Tensors**:
  ```python
  print(x.shape)       # Tensor shape
  print(x.dtype)       # Data type
  print(x.ndimension())  # Number of dimensions
  ```

---

## 🔄 **Tensor Manipulations**

- **Reshape**: Change tensor shape.
  ```python
  x = torch.arange(12).reshape(3, 4)  # (3, 4)
  ```

- **Squeeze/Unsqueeze**: Remove or add dimensions of size 1.
  ```python
  x = x.unsqueeze(0)  # Add a dimension
  x = x.squeeze()     # Remove dimensions of size 1
  ```

- **View**: Similar to `reshape` but requires contiguous memory.
  ```python
  x = x.view(3, 4)
  ```

---

## 🔢 **Matrix Operations**

1. **Matrix Multiplication**:
   ```python
   torch.mm(A, B)       # 2D matrix multiplication
   torch.matmul(A, B)   # Generalized matrix multiplication
   ```

2. **Batch Matrix Multiplication**:
   ```python
   torch.bmm(A, B)  # For 3D tensors
   ```

---

## 🔍 **Tensor Indexing**

- **Basic Indexing**:
  ```python
  x = torch.tensor([[1, 2], [3, 4]])
  print(x[0, 1])  # Access element
  ```

- **Advanced Indexing**:
  ```python
  indices = torch.tensor([0, 1])
  x_selected = torch.index_select(x, 0, indices)
  ```

---

## ↔️ **Broadcasting and Expanding**

1. **Expand**:
   Expands dimensions without copying data.
   ```python
   x = torch.tensor([1, 2, 3])
   x_expanded = x.unsqueeze(0).expand(2, 3)
   ```

2. **Repeat**:
   Repeats data along dimensions.
   ```python
   x_repeated = x.repeat(2, 1)
   ```

---

## ➗ **Tensor Reduction**

- **Sum**:
  ```python
  torch.sum(x, dim=0)
  ```

- **Mean**:
  ```python
  torch.mean(x.float(), dim=1)
  ```

- **Product**:
  ```python
  torch.prod(x, dim=1)
  ```

---

## 📐 **Linear Algebra**

- **Matrix Inverse**:
  ```python
  torch.inverse(A)
  ```

- **SVD**:
  ```python
  torch.svd(A)
  ```

- **Eigenvalues**:
  ```python
  torch.eig(A, eigenvectors=True)
  ```

---

## ✅ **Comparison Operations**

1. **Element-wise Comparison**:
   ```python
   torch.eq(A, B)  # Equal
   torch.gt(A, 2)  # Greater than
   ```

2. **Conditional Selection**:
   ```python
   torch.where(A > 2, A, B)
   ```

---

## 📖 **Additional Resources**

- [PyTorch Official Documentation](https://pytorch.org/docs/)
- [Deep Learning with PyTorch: A 60 Minute Blitz](https://pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html)
- [PyTorch Forums](https://discuss.pytorch.org/)

---

## 🤝 **Contributing**

Feel free to submit pull requests, report issues, or suggest improvements to this repository. Let’s make learning PyTorch easier for everyone!

---

## 📜 **License**

This repository is licensed under the MIT License. Feel free to use and modify it for educational purposes.
