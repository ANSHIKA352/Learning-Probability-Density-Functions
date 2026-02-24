# Learning Probability Density Function using GAN
### NO₂ Air Quality Data – Generative Modeling Assignment

---

## Objective

The objective of this assignment is to learn the unknown probability density function (PDF) of a transformed NO₂ concentration variable using a Generative Adversarial Network (GAN).

No parametric distribution (Gaussian, exponential, etc.) is assumed.  
The GAN learns the distribution purely from data samples.

---

## Dataset

- Dataset: India Air Quality Data
- Feature used: NO₂ concentration
- Missing values removed
- Extreme outliers (above 99th percentile) removed

---

## Data Transformation

Each NO₂ value \( x \) was transformed using:

z = x + a_r sin(b_r x)

Where:

a_r = 0.5 (r mod 7)  
b_r = 0.3 (r mod 5 + 1)

For roll number:

102303042

Computed parameters:

a_r = 3.0  
b_r = 0.9  

Final transformation applied:

z = x + 3 sin(0.9x)

This transformation introduces nonlinear oscillatory behavior and makes the distribution multimodal.

---

## GAN Architecture

### Generator
- Input: Noise ~ N(0,1)
- Fully connected network
- Output activation: Sigmoid (data normalized)

Layers:
- Linear(1 → 32)
- ReLU
- Linear(32 → 64)
- ReLU
- Linear(64 → 1)
- Sigmoid

---

### Discriminator
- Input: Real or generated sample
- Output: Probability (real/fake)

Layers:
- Linear(1 → 64)
- LeakyReLU
- Linear(64 → 32)
- LeakyReLU
- Linear(32 → 1)
- Sigmoid

---

## Training Details

- Loss Function: Binary Cross Entropy (BCELoss)
- Optimizer: Adam
- Learning Rate: 0.0001
- Epochs: 5000
- Batch Size: 128
- Data normalized using MinMaxScaler

---

## PDF Estimation

After training:

1. 10,000 samples were generated from the trained Generator.
2. Samples were inverse-transformed to original scale.
3. Kernel Density Estimation (KDE) was used to estimate the learned PDF.
4. True PDF and Learned PDF were compared.

---

## Results

The GAN successfully learned the overall structure of the transformed distribution.

### Observations

- Major modes were captured.
- Fine oscillatory ripples were partially smoothed.
- Slight underestimation in extreme tails.
- No severe mode collapse observed.
- Training remained stable.

---

## PDF Comparison Plot

![PDF Comparison](outputs/pdf_plot.png)

---

## Libraries Used

- PyTorch
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn

---

## Conclusion

This project demonstrates that a GAN can approximate complex multimodal distributions using only sample data without assuming any analytical probability density function.
