# Linear Probes for Adversarial Attack Detection

This repository contains a simple proof-of-concept study on whether adversarial examples can be distinguished from clean inputs using **linear probes** trained on intermediate activations of deep neural networks.

## Overview

Adversarial examples are small perturbations to inputs that fool neural networks while remaining almost imperceptible to humans. This project investigates:

- Are activations from adversarial vs. clean inputs fundamentally different?
- Are these differences **linearly separable** in activation space?
- Can we use linear probes trained on activations to detect adversarial attacks?

The experiments focus mainly on the **Fast Gradient Sign Method (FGSM)**, with additional tests applying Gaussian blur to study the role of high-frequency components.

## Methodology

1. **Generate adversarial images**  
   - Attacks: FGSM (optionally combined with Gaussian blur).  
   - Dataset: TinyImageNet.  
   - Example: 5000 adversarial samples generated for each model.

2. **Extract activations**  
   - Models: ResNet family (e.g., ResNet34, ResNet101, ResNet152).  
   - Activations are averaged across spatial dimensions for each layer.

3. **Train linear probes**  
   - A simple linear classifier is trained to predict whether activations come from adversarial (1) or clean (0) inputs.  
   - Probe accuracy is measured layer by layer.

## Key Findings

- **Middle layers perform best**: Probes reach **80–90% test accuracy** when trained on FGSM adversarial examples, especially at early-to-middle layers.  
- **Not a complete solution**: The probes seem tuned to detect **high-frequency artifacts** from FGSM.  
- **Gaussian blur test**:  
  - Applying blur reduces probe accuracy significantly, but not to random chance.  
  - The adversarial attack remains effective (misclassification rate above 98%).  
  - Early-to-middle layers still retain detectable signals, suggesting more than just “high frequency noise detection.”  
- **Hypothesis**:  
  - Early/middle layers capture chaotic, inconsistent activations that are easy to separate.  
  - Final layers resemble consistent representations of the adversarial target class, making separation harder.

## Next Steps

- Compare hypotheses for why later layers are harder to separate.  
- Test on other attack methods and varying attack strengths (ϵ).  
- Explore whether attacks can be crafted to fool both the model and the probes.  
- Investigate how adversarial training reshapes internal representations.

## Running the Code

All experiments are contained in the Jupyter notebook [`experiments.ipynb`](./experiments.ipynb).  
To reproduce results, set the following parameters at the top of the second cell:

```python
modelname  = "resnet34"      # e.g., "resnet18", "resnet34", "resnet50", "resnet101", "resnet152"
attackname = "fgsm+blur"     # or "fgsm"
blursigma  = 1.0             # standard deviation of Gaussian blur
epsilon    = 0.02            # FGSM attack strength (epsilon)