# Neuromorphic Engineering 🧠

A repository of computational neuroscience models and simulations exploring biologically inspired neural systems, with an emphasis on **spiking neural networks**, **point neuron dynamics**, and **neuromorphic computation using Nengo**.

## Brain-Inspired Computing Architectures: Scientific, Architectural, and Algorithmic Perspectives  

Before we dive into the project code and artifacts, let's explore the three common perspective for brain-inspired computing.

### The Scientific Perspective

Understanding the brain requires grappling with one of the most profound challenges in science: how simple components, such as neurons, interact to form complex behaviors and cognition. A key concept in this endeavor is **emergent behavior**—a phenomenon where group-level behaviors arise from simple, local interactions among system components. This behavior is not present in individual units but emerges only at the system level.

A fascinating example of emergent behavior can be seen in **synchronous fireflies**. These fireflies, particularly the *Photinus carolinus* species, synchronize their flashing patterns without centralized control. Each firefly follows simple rules, but together they produce a stunning collective rhythm. Studies on this phenomenon, such as John Buck’s seminal work, illustrate how complex biological coordination can arise from local interactions.

In brain research, emergent behavior challenges the reductionist approach of studying neurons in isolation. Consciousness, for instance, likely emerges from the intricate interactions between neural populations rather than from individual neurons. Thus, studying the brain demands a **multi-layered abstraction approach**, addressing behavior from the molecular to the cognitive level.

This leads to one of the core challenges in neuroscience: deciding which abstraction level to model. Since the true origin of consciousness and cognition remains unknown, one proposed solution is to **build an artificial brain**. By creating systems that mimic brain function at different levels, we can experiment with and observe emergent behaviors that mirror natural cognition. This follows the famous principle of Richard Feynman:  
> “What I cannot create, I do not understand.”

To address this, researchers employ two dominant modeling strategies: **bottom-up** and **top-down** approaches.

#### Bottom-Up vs. Top-Down Modeling

- **Bottom-Up Modeling** starts from low-level abstractions - such as neurons and synapses - and builds upward. This method offers high **explanatory power** and **biological realism**, making it ideal for studying the origin of specific emergent behaviors. However, it is computationally expensive and often fails to scale to complex behaviors such as intelligence. For example, IBM’s 2009 simulation “The Cat is Out of the Bag” illustrates the limits of this approach in practice.

- **Top-Down Modeling** begins with high-level behavior or function and designs lower-level components to achieve it. This method excels at modeling **emergent phenomena** efficiently and is more flexible in adjusting internal components based on external goals—similar to how artificial neural networks self-adjust during training. Its downside is lower explanatory power, as it provides fewer insights into how those lower-level behaviors develop naturally.

In practice, both approaches are complementary. While bottom-up models deepen our understanding, top-down models provide efficient simulations of intelligence. Importantly, both approaches require vast computational resources—highlighting the need for **neuromorphic computing** to model brain-like systems effectively.

### The Computer Architecture Perspective

The historical trajectory of computing is deeply influenced by **Dennard Scaling** and **Amdahl’s Law**, two principles that once drove exponential performance gains.

- **Dennard Scaling** states that as transistors become smaller, their power density remains constant, allowing more transistors to be packed into a chip without increasing heat or energy consumption. This enabled the continuation of **Moore’s Law**, which predicted a doubling of performance every two years.

- **Amdahl’s Law**, on the other hand, highlights the **limits of parallel computing**. It shows that the speedup of a task through parallelism is constrained by the portion of the task that must remain sequential. Even if 80% of a task is parallelizable, the theoretical maximum speedup is only 5×.

Together, these principles guided the design of CPUs and parallel architectures for decades. However, today’s reality is starkly different. **Moore’s Law is considered dead**, not because innovation has stopped, but because physical and economic constraints have stalled the exponential growth in transistor performance.

#### Why Moore’s Law Has Stalled

Several factors have contributed to the slowdown:

- **Physical limits** in transistor miniaturization result in electron trapping and quantum effects.
- **3D transistors**, although more efficient, are significantly more expensive.
- **Heat dissipation challenges** prevent higher clock speeds.
- **Parallelism limitations** due to Amdahl’s Law restrict performance scaling.
- **Energy consumption** in GPUs and CPUs (250W–300W) necessitates sophisticated cooling.
- **Empirical evidence** shows a decline in annual performance gains:
  - 1986–2002: ~52%
  - 2003–2010: ~25%
  - 2011–2015: ~12%
  - 2015–Present: ~3.5%

These realities underscore that traditional methods - smaller, faster, more transistors - are no longer viable.

#### A Counterpoint: Moore’s Law is Evolving

Despite these limitations, some researchers argue that **Moore’s Law is evolving rather than dying**. In the article *"AI, Native Supercomputing and the Revival of Moore's Law"*, the author suggests that domain-specific architectures, especially for AI and deep learning, are enabling new forms of exponential growth.

Specialized AI chips, such as tensor processing units (TPUs), exploit **massive parallelism** for matrix operations, which are the foundation of modern deep learning. These chips decouple performance gains from traditional transistor scaling, offering an alternate path for progress.

In this view, Moore’s Law lives on—not in hardware miniaturization, but in **architectural innovation**. As Prof. Kwabena Boahen emphasizes, to maintain progress, we must pursue fundamentally different hardware models, such as those found in **neuromorphic computing**.

### The Algorithmic Perspective

The algorithmic domain bridges the gap between models of the brain and functional artificial intelligence. At its core, this includes **artificial neural networks (ANNs)** and their biologically inspired counterpart, **spiking neural networks (SNNs)**.

#### ANN vs. SNN

- **ANNs** consist of layers of artificial neurons that process information using **continuous and differentiable values**. These systems are trained with backpropagation and are highly effective in pattern recognition, natural language processing, and vision tasks.

- **SNNs**, in contrast, communicate using **discrete spikes** over time, more closely resembling the behavior of real neurons. They are **event-driven**, energy-efficient, and well-suited for real-time processing in noisy environments.

| Feature                  | ANN                                | SNN                                |
|--------------------------|-------------------------------------|-------------------------------------|
| Data representation      | Continuous, differentiable          | Discrete spikes, temporal coding    |
| Communication            | Synchronous                        | Asynchronous, event-driven          |
| Biological realism       | Low                                | High                                |
| Use cases                | Classification, NLP                | Real-time, noisy, energy-efficient  |
| Hardware needs           | GPUs/TPUs                          | Neuromorphic chips (e.g., Loihi)    |

While ANNs dominate today's AI applications, SNNs represent a promising path for **energy-efficient, brain-like computing**. However, they require specialized hardware and training algorithms that are still maturing.

#### Additional Architectures: CNNs, RNNs, and BNNs

- **BNNs (Biological Neural Networks)** refer to real neuronal systems, such as those found in the human gut-brain axis (ENS). Research on "Brain in a Dish" experiments demonstrates the ability to simulate parts of these networks outside the human body.

- **CNNs (Convolutional Neural Networks)** are designed for processing spatial data. They are widely used in image recognition, such as in safety monitoring systems on construction sites using YOLOv8.

- **RNNs (Recurrent Neural Networks)** are tailored for sequential data. They process time-series information and are fundamental in applications like machine translation. Notably, the *Sequence to Sequence Learning with Neural Networks* paper by Sutskever et al. (2014) introduced groundbreaking methods for sequence prediction.

### Conclusion

The study of brain-inspired computing requires a **multi-disciplinary approach** spanning science, architecture, and algorithms. Each perspective brings unique insights:

- The scientific perspective emphasizes **emergent behavior** and abstraction layers.
- The architectural perspective grapples with **hardware limitations** and explores new models like **neuromorphic chips**.
- The algorithmic perspective focuses on **mimicking brain function** through ANNs and SNNs.

Together, they lay the groundwork for a future where artificial systems not only simulate intelligence but do so with the **efficiency, robustness, and adaptability of the human brain**.




## Project Structure

```
neuromorphic-engineering/
├── morphologically_detailed_neuron_models/
├── point_neuronal_dynamic_models/
├── snn/
├── nengo/
└── README.md
```


## Morphologically Detailed Neuron Models

**Location**: `morphologically_detailed_neuron_models/`

This module contains implementations and simulations of morphologically accurate neuron models using:

* `the_compartmental_model.py`: Compartmental modeling of neurons.
* `the_cable_equation.py`: Classical cable equation for dendritic voltage propagation.
* `the_cable_equation_2.py`: A variant of the cable equation implementation.

## Point Neuronal Dynamic Models

**Location**: `point_neuronal_dynamic_models/`

Focused on **single-compartment models** of neural spiking dynamics:

* `LIF.py`, `LIF-HW1.py`, `LIF-HW2.py`: Leaky Integrate-and-Fire neuron models.
* `Izhikevich.py`, `Izhikevich_ref.py`, `Izhikevich-HW.py`: Izhikevich neuron types with different firing regimes.
* `HnH.py`: Hodgkin–Huxley model implementation.
* `*_plots/`: Visualization of voltage traces and parameter variations for different models.

## Nengo Simulations

**Location**: `nengo/`

This submodule includes a wide variety of **simulations using the Nengo framework**, illustrating key concepts of neuromorphic computing and the Neural Engineering Framework (NEF).

### Highlights

#### Representation

* `representation.py`: LIF vs. Rectified Linear tuning curves, ensemble decoding, high-dimensional analysis.

#### Transformation

* `transformation.py`: Decoder-based function transformations (e.g., `sin(x)` to `sin²(x)`, vector sums, multiplication, gating).

#### Dynamics

* `dynamics.py`: Recurrent connections modeling functions like `f(x)=x+1`, `f(x)=x²`, `f(x)=-x`, integrators, oscillators, and Lorenz attractor.

#### Learning and Adaptation

* `pes.py`: Online learning using the PES rule in a simple communication channel.
* `pavlovian.py`: Classical conditioning via Hebbian and PES learning mechanisms.

#### Interactive Experiments

**Location**: `nengo/HW/`

* Tuning curves (`tuning_curves.py`)
* Function approximations (`non_linear.py`, `transformation_hw.py`)
* Ring attractors (`ring_attractor.py`, `Ring_Attractor.ipynb`)
* High-dimensional stimuli (`high_dim_stim.py`)
* Performance metrics & visualizations (`accuracy.py`, `radius.py`)

## Spiking Neural Networks (SNN)

**Location**: `snn/`

Advanced simulations of SNNs and learning algorithms.

* `diffrentiable_lif.py`: Tuning curves for LIF neurons with differentiable nonlinearities.
* `mnist.py`: A spiking convolutional neural network (SNN) for MNIST classification using **NengoDL**.
* `pavlovian.py`: Pavlovian conditioning using ensembles and learning rules.
* `pes.py`: Implementation of PES learning on a feedforward task.

## Visualizations

The `plots/` folder (inside `nengo/HW/`) includes a rich collection of visual outputs generated by the simulations, such as:

* Tuning curves of various neuron ensembles
* Function transformations over different neuron counts
* Dynamic attractor states
* Accuracy over high-dimensional input spaces

## Requirements

* Python ≥ 3.8
* `nengo`
* `nengo-dl`
* `tensorflow`
* `matplotlib`
* `numpy`
* `seaborn`
* `scipy`

Install dependencies:

```bash
pip install -r requirements.txt
```

*Note: Some simulations require GPU acceleration for NengoDL to efficiently run MNIST classification.*

## References

* Neural Engineering Framework (NEF)
* Nengo: [https://www.nengo.ai](https://www.nengo.ai)
* NengoDL: [https://www.nengo.ai/nengo-dl](https://www.nengo.ai/nengo-dl)
* Izhikevich, E. M. (2003). Simple model of spiking neurons.

## Note

This project is part of a neuromorphic computing exploration portfolio for academic and research use.
