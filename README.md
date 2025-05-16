# DIP-GUI-Toolkit

A Python-based image processing toolkit with a modern GUI built using Tkinter and ttkbootstrap. Designed for educational and experimental use, this tool supports a wide range of grayscale and RGB operations, from basic point manipulation to advanced edge detection and segmentation.

![image](https://github.com/user-attachments/assets/9faf37c7-dff4-42e5-9c4e-b6f1ada7f3a8)

## 🚀 Features

- ✅ Load and display grayscale or RGB images
- 🎛️ Basic operations: Brightness, Contrast (gamma, stretching, matching)
- ⚡ Point-wise operations: Addition, Subtraction, Division, Complement
- 🎨 Color operations: Channel elimination, intensity changes, channel swap
- 📊 Histogram analysis: Equalization, Stretching
- 🔍 Filtering:
  - Linear: Average, Gaussian, Laplacian
  - Non-linear: Median, Min, Max, Outlier, Most frequent
- 🧹 Noise removal: Salt-and-pepper, Gaussian noise with averaging
- ✂️ Segmentation: Global, Automatic, Adaptive thresholding
- 🧭 Edge detection: Sobel, Prewitt, Roberts, Laplacian (with kernel selector)
- 🧪 Test Mode: Add your own custom tests and experimental modules
- 🔄 Undo feature for previous operations

## 📦 Installation

```bash
git clone https://github.com/yourusername/DIP-GUI-Toolkit.git
cd DIP-GUI-Toolkit
pip install -r requirements.txt
```

### Requirements

- Python 3.8+
- ttkbootstrap
- Pillow
- numpy
- OpenCV (cv2)
- pandas

Install all dependencies:

```bash
pip install -r requirements.txt
```

## ▶️ Usage

```bash
python main.py
```

- Use the **Upload Image** button to load an image.
- Select an operation from the **Basic Operations** dropdown.
- Adjust parameters (e.g., sliders, kernel sizes) and click the corresponding **Apply** button.
- Use **Undo** to revert to the previous step.
- Save processed image or histogram if needed.

## 🛠️ Project Structure

```
├── main.py                  # GUI layout and control flow
├── operations_basic.py      # All grayscale and filter operations
├── operations_color.py      # Color-specific operations
├── image_io.py              # Image loading/saving/display functions
├── requirements.txt
└── README.md
```

## 🧠 Educational Use

This project is ideal for students learning digital image processing (DIP), offering real-time interaction with common filters, transformations, and enhancement techniques.

## 📜 License

MIT License — feel free to use and extend with credit.

---

**Built with ❤️ for learning, experimenting, and mastering image processing. **
