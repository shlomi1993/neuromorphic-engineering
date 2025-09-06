# Neuromorphic Engineering 🧠

This repository explores **brain-inspired computing architectures** and contains computational neuroscience models and simulations with a focus on:

- **Point neuron dynamics**
- **Spiking Neural Networks (SNNs)**
- **Neuromorphic computation simulation using Nengo**

## 1. Overview: Representation, Transformation, and Dynamics in the NEF

<img width="1425" height="715" alt="Screenshot" src="https://github.com/user-attachments/assets/a95a3978-6552-4c14-9c49-f77e69ad4dd7" />

The figure above illustrates NEF's core principles across four stages:

1. **Encoding** – Spike trains from 8 neurons represent a ramp signal.
2. **Decoding** – Spikes are decoded to reconstruct inputs and compute $\cos(\text{input})$.
3. **Transformation** – Computation of $\sin(x)$, $-x$, and $x^2$ via spiking.
4. **Dynamics** – A spiking 2D oscillator mimics circular behavior.

## 2. Scientific, Architectural, and Algorithmic Perspectives

<details>
<summary>Click to expand</summary>

### 2.1 The Scientific Perspective

Understanding the brain requires grappling with one of the most profound challenges in science: how simple components, such as neurons, interact to form complex behaviors and cognition. A key concept in this endeavor is **emergent behavior**—a phenomenon where group-level behaviors arise from simple, local interactions among system components. This behavior is not present in individual units but emerges only at the system level.

A fascinating example of emergent behavior can be seen in **synchronous fireflies**. These fireflies, particularly the *Photinus carolinus* species, synchronize their flashing patterns without centralized control. Each firefly follows simple rules, but together they produce a stunning collective rhythm. Studies on this phenomenon, such as John Buck’s seminal work, illustrate how complex biological coordination can arise from local interactions.

In brain research, emergent behavior challenges the reductionist approach of studying neurons in isolation. Consciousness, for instance, likely emerges from the intricate interactions between neural populations rather than from individual neurons. Thus, studying the brain demands a **multi-layered abstraction approach**, addressing behavior from the molecular to the cognitive level.

This leads to one of the core challenges in neuroscience: deciding which abstraction level to model. Since the true origin of consciousness and cognition remains unknown, one proposed solution is to **build an artificial brain**. By creating systems that mimic brain function at different levels, we can experiment with and observe emergent behaviors that mirror natural cognition. This follows the famous principle of Richard Feynman:  
> “What I cannot create, I do not understand.”

To address this, researchers employ two dominant modeling strategies: **bottom-up** and **top-down** approaches.

#### 2.1.1 Bottom-Up vs. Top-Down Modeling

- **Bottom-Up Modeling** starts from low-level abstractions - such as neurons and synapses - and builds upward. This method offers high **explanatory power** and **biological realism**, making it ideal for studying the origin of specific emergent behaviors. However, it is computationally expensive and often fails to scale to complex behaviors such as intelligence. For example, IBM’s 2009 simulation “The Cat is Out of the Bag” illustrates the limits of this approach in practice.

- **Top-Down Modeling** begins with high-level behavior or function and designs lower-level components to achieve it. This method excels at modeling **emergent phenomena** efficiently and is more flexible in adjusting internal components based on external goals—similar to how artificial neural networks self-adjust during training. Its downside is lower explanatory power, as it provides fewer insights into how those lower-level behaviors develop naturally.

In practice, both approaches are complementary. While bottom-up models deepen our understanding, top-down models provide efficient simulations of intelligence. Importantly, both approaches require vast computational resources—highlighting the need for **neuromorphic computing** to model brain-like systems effectively.

### 2.2 The Computer Architecture Perspective

The historical trajectory of computing is deeply influenced by **Dennard Scaling** and **Amdahl’s Law**, two principles that once drove exponential performance gains.

- **Dennard Scaling** states that as transistors become smaller, their power density remains constant, allowing more transistors to be packed into a chip without increasing heat or energy consumption. This enabled the continuation of **Moore’s Law**, which predicted a doubling of performance every two years.

- **Amdahl’s Law**, on the other hand, highlights the **limits of parallel computing**. It shows that the speedup of a task through parallelism is constrained by the portion of the task that must remain sequential. Even if 80% of a task is parallelizable, the theoretical maximum speedup is only 5×.

Together, these principles guided the design of CPUs and parallel architectures for decades. However, today’s reality is starkly different. **Moore’s Law is considered dead**, not because innovation has stopped, but because physical and economic constraints have stalled the exponential growth in transistor performance.

#### 2.2.1 Why Moore’s Law Has Stalled

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

#### 2.2.2 A Counterpoint: Moore’s Law is Evolving

Despite these limitations, some researchers argue that **Moore’s Law is evolving rather than dying**. In the article *"AI, Native Supercomputing and the Revival of Moore's Law"*, the author suggests that domain-specific architectures, especially for AI and deep learning, are enabling new forms of exponential growth.

Specialized AI chips, such as tensor processing units (TPUs), exploit **massive parallelism** for matrix operations, which are the foundation of modern deep learning. These chips decouple performance gains from traditional transistor scaling, offering an alternate path for progress.

In this view, Moore’s Law lives on—not in hardware miniaturization, but in **architectural innovation**. As Prof. Kwabena Boahen emphasizes, to maintain progress, we must pursue fundamentally different hardware models, such as those found in **neuromorphic computing**.

### 2.3 The Algorithmic Perspective

The algorithmic domain bridges the gap between models of the brain and functional artificial intelligence. At its core, this includes **artificial neural networks (ANNs)** and their biologically inspired counterpart, **spiking neural networks (SNNs)**.

#### 2.3.1 ANN vs. SNN

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

#### 2.3.2 Additional Architectures: CNNs, RNNs, and BNNs

- **BNNs (Biological Neural Networks)** refer to real neuronal systems, such as those found in the human gut-brain axis (ENS). Research on "Brain in a Dish" experiments demonstrates the ability to simulate parts of these networks outside the human body.

- **CNNs (Convolutional Neural Networks)** are designed for processing spatial data. They are widely used in image recognition, such as in safety monitoring systems on construction sites using YOLOv8.

- **RNNs (Recurrent Neural Networks)** are tailored for sequential data. They process time-series information and are fundamental in applications like machine translation. Notably, the *Sequence to Sequence Learning with Neural Networks* paper by Sutskever et al. (2014) introduced groundbreaking methods for sequence prediction.

### 2.4 Conclusion

The study of brain-inspired computing requires a **multi-disciplinary approach** spanning science, architecture, and algorithms. Each perspective brings unique insights:

- The scientific perspective emphasizes **emergent behavior** and abstraction layers.
- The architectural perspective grapples with **hardware limitations** and explores new models like **neuromorphic chips**.
- The algorithmic perspective focuses on **mimicking brain function** through ANNs and SNNs.

Together, they lay the groundwork for a future where artificial systems not only simulate intelligence but do so with the **efficiency, robustness, and adaptability of the human brain**.

</details>

## 3. The Neuron Model

<details>
<summary>Click to expand</summary>

Now, let's explore three iconic neuron models.

### 3.1 Leaky Integrate-and-Fire (LIF) Model

#### 3.1.1 I-F Curves and the Effect of Membrane Time Constant $τ$

The Leaky Integrate-and-Fire (LIF) model is an electrical-mathematical model that simulates point neuron behavior. It includes a capacitor, representing ion separation across the membrane, and a resistor, representing membrane permeability. In the absence of stimulation, the capacitor voltage exponentially decays ("leaks") to a resting potential through the resistor.

<p align="center"><img width="121" height="160" src="https://github.com/user-attachments/assets/60cc97f0-0a0c-4b5f-a74a-dad4bfc97d38" /></p>

The current conservation:

$I(t)=I_R(t) + I_C(t)$

Leads to the model equation:

$τ · dV(t)/dt=R · I(t) - V(t)$

Where:
- $τ$ is the membrane time constant, defined by $τ=RC$.
- $V(t)$ is the membrane potential.

Using the iterative method:

$u_∞(i)=u_{rest} + R · I(i) u(i+1)=u_∞ + (u(i) - u_∞) · e^(-dt/τ)$

#### 3.1.2 Simulation Parameters:
- $u_{rest}=-70 mV$
- $V_{th}=-40 mV$
- $R=1 kΩ$
- $dt=0.1 ms$
- $T=50 ms$

We vary the current:

$I(t_i)=dI · i$ where $dI=0.5 µA$

##### 3.1.2.1 Observations for Different $τ$ Values:

$τ=0.01$: First spike occurs at $~94.5 µA$, initial firing rate $≈ 0.18 Hz$
<p align="center"><img width="740" height="408" alt="image" src="https://github.com/user-attachments/assets/f8fb2c2c-51c6-420d-b0be-602f86dd5341" /></p>

$τ=0.02$: Shift in the curve, first spike at $~126.5 µA$, frequency $≈ 0.1351 Hz$
<p align="center"><img width="736" height="403" alt="image" src="https://github.com/user-attachments/assets/d485cc79-7720-4691-a55a-57120a993d13" /></p>

$τ=0.03$: First spike at $~151 µA$, frequency $≈ 0.1124 Hz$
<p align="center"><img width="740" height="402" alt="image" src="https://github.com/user-attachments/assets/1dcc2e0a-d99b-45c6-a1fa-8c118d23145b" /></p>

Larger $τ$ values result in lower firing frequencies for the same current due to slower membrane potential buildup.

#### 3.1.2.2 V-T Curves for Different Thresholds

Flat current input: $I(t)=0.0001 A$

Model parameters:
- $R=1 kΩ$
- $C=5 µF ⇒ τ=RC=0.005$
- $dt=0.1 ms$
- $T=50 ms$

Threshold values: $V_{th} ∈ {-70 mV, -30 mV, 10 mV}$

##### 3.1.2.3 Results:

$V_{th}=-70 mV$: Immediate firing, stable periodic spikes
<p align="center"><img width="731" height="405" alt="image" src="https://github.com/user-attachments/assets/988d3ee9-6a79-4619-9114-1213df519b11" /></p>

$V_{th}=-30 mV$: First spike at $t=2.7 ms$, periodic
<p align="center"><img width="733" height="400" alt="image" src="https://github.com/user-attachments/assets/aa754770-978b-4873-a9f7-09240f057913" /></p>

$V_{th}=10 mV$: First spike at $t=8.2 ms$, lower frequency
<p align="center"><img width="734" height="405" alt="image" src="https://github.com/user-attachments/assets/bb3f9c0a-ec40-498e-a25c-7ba3c729f7d5" /></p>

As $V_{th}$ increases, firing starts later and occurs less frequently.

#### 3.1.3 Time to Reach Threshold

Using:

$t_{th}=-τ · ln((V_{th} - u_{rest})/(R · I_0))$

Substituting values:
- For $V_{th}=-70 mV$: $t_{th}=0 ms$
- For $V_{th}=-30 mV$: $t_{th}≈2.6 ms$
- For $V_{th}=10 mV$: $t_{th}≈8.1 ms$

Higher threshold values result in increased time to spike.

### 3.2 Izhikevich Model

#### 3.2.1 Eight Firing Modes

Using the guide-provided code with:
- $dt=0.1 ms$
- Three input types: step, step with pulse, negative step

Firing modes replicated:
- Regular Spiking (RS)
- Intrinsically Bursting (IB)
- Chattering (CH)
- Fast Spiking (FS)
- Low-Threshold Spiking (LTS)
- Resonator (RZ)
- Thalamo-Cortical (TC) with $v_0=-63$
- TC with $v_0=-87$

### 3.2.2 Mode Characteristics

Model equations:

$v'=0.04v^2 + 5v + 140 - u + I u'=a(bv - u)$

After spike ($v >= 30$): $v ← c$, $u ← u + d$

**RS: Spike frequency adaptation due to low $c$ and high $d$**
<p align="center"><img width="624" height="346" alt="image" src="https://github.com/user-attachments/assets/34cfd91d-6834-4569-84a5-508106245e20" /></p>

**IB: Initial burst, then slower firing ($c=-55$, $d=4$)**
<p align="center"><img width="622" height="344" alt="image" src="https://github.com/user-attachments/assets/72443435-4a00-4294-8e8d-6c193c4906a3" /></p>

**CH: High-frequency bursts ($c=-50$, $d=2$)**
<p align="center"><img width="623" height="340" alt="image" src="https://github.com/user-attachments/assets/5bf87727-2c0c-49db-8ec0-fec210106f69" /></p>

**FS: High constant firing due to $a=0.1$**
<p align="center"><img width="621" height="341" alt="image" src="https://github.com/user-attachments/assets/5f0288fe-8362-4ecd-b7aa-459dd31f23e1" /></p>

**LTS: Low threshold due to $b=0.25$**
<p align="center"><img width="624" height="346" alt="image" src="https://github.com/user-attachments/assets/73065e85-efd6-4ec1-a64b-8fe8ff6539bb" /></p>

**RZ: Rebound spikes due to $a=0.1$, $b=0.26$**
<p align="center"><img width="620" height="338" alt="image" src="https://github.com/user-attachments/assets/fe4d6156-9a0b-4d64-a45c-3976e50b6f05" /></p>

**TC (-63): Gradual spike adaptation**
<p align="center"><img width="619" height="342" alt="image" src="https://github.com/user-attachments/assets/01422c90-41d8-410f-906f-001fa2faae60" /></p>

**TC (-87): Burst after inhibitory input**
<p align="center"><img width="629" height="338" alt="image" src="https://github.com/user-attachments/assets/4879f393-1460-46b7-8f64-1e4d56a88588" /></p>

### Hodgkin-Huxley Model
<p align="center"><img width="186" height="161" alt="image" src="https://github.com/user-attachments/assets/ee85000c-8559-4afa-addc-bac8a5b690b1" /></p>

#### 3.2.3 Significance of $E_K$, $E_{Na}$, $E_{leak}$

Model equation:

$I(t)=C_m · dV_m/dt + I_K + I_Na + I_leak$

Where:
- $I_K=g_K · n^4 · (V_m - V_K)$
- $I_{Na}=g_{Na} · m^3 · h · (V_m - V_{Na})$
- $I_{leak}=g_{leak} · (V_m - V_{leak})$

Gating variables $m$, $n$, $h$ depend on voltage via $α$ and $β$ functions. The voltages $E_K$, $E_{Na}$, $E_{leak}$ determine the ion flow direction.

#### 3.2.4 Effect of $E_K$, $E_{Na}$, $E_{leak}$ on Spikes

Default: $E_{Na}=115$, $E_K=-12$, $E_{leak}=10.6$
<p align="center"><img width="748" height="405" alt="image" src="https://github.com/user-attachments/assets/7b73d01c-585c-4aeb-bc0f-6e5466447da0" /></p>

#### 3.2.5 Variations:
$E_{Na}=180$: Increased spike height
<p align="center"><img width="746" height="397" alt="image" src="https://github.com/user-attachments/assets/42a17de6-06c5-424e-b503-8dd1dbbe43bc" /></p>

$E_K=10$: Reduced spike height
<p align="center"><img width="739" height="397" alt="image" src="https://github.com/user-attachments/assets/76a35e03-5c5e-4627-b7ba-ab1c4032002d" /></p>

$E_{leak}=0$: Reduced spike frequency
<p align="center"><img width="737" height="402" alt="image" src="https://github.com/user-attachments/assets/45784ee7-e5d5-4555-8166-335475d33854" /></p>

The model highlights how spike dynamics emerge from ionic mechanisms rather than explicit spike logic, illustrating its biological plausibility.

</details>


## 4. The Neural Engineering Framework

<details>
<summary>Click to expand</summary>

### 4.1 Data Representation via NEF

The **Neural Engineering Framework (NEF)** is a computational framework designed for modeling neural systems at scale. Unlike traditional, bottom-up neural modeling, NEF adopts a top-down approach: high-level specifications of a neural network determine its low-level structure. This makes it possible to model highly complex dynamical systems.

NEF relies on three foundational principles:

1. **Representation** – Uses ensembles of spiking neurons to encode and decode continuous or non-linear signals in a distributed manner.
2. **Transformation/Computation** – Implements both linear and non-linear functions through weighted synaptic connections between neural ensembles.
3. **Dynamics** – Creates feedback connections within ensembles to simulate dynamic behaviors, such as working memory.

These principles are implemented in **Nengo**, a Python library that provides convenient APIs to define:

- **Node** – A non-neural input/output interface that injects external data into the neural model or processes its outputs. Nodes are often used for providing stimuli or controlling the simulation environment.
  
- **Ensemble** – A population of neurons that encode and decode signals via tuning curves, the core computational units in NEF. Each ensemble represents vectors, computes functions, or detects errors.
  
- **Connection** – Connects nodes or ensembles, specifying weighted synaptic pathways. Connections define how pre-synaptic neural activity transforms into post-synaptic responses.

Together, these components allow researchers to build large-scale, biologically plausible neural simulations with clear abstractions for how information is represented, processed, and flowed.

### 4.2 Basis Functions and Their Importance

In NEF, **basis functions** serve as essential tools for representing and transforming information within neural networks. They parallel the concept of basis vectors in linear algebra but apply it to function spaces.

Neuron tuning curves in NEF act as a full basis for the functional space that the neurons can compute. Because each neuron responds with a characteristic tuning curve, an ensemble can represent a wide range of inputs via weighted summation of activity patterns.

#### 4.2.1 Why basis functions matter:

- **Efficient representation** of continuous signals using discrete spiking neurons, supporting high-dimensional and non-linear data.
- **Neural computation**, enabling linear and non-linear transformations by mapping inputs to neural activities and decoding outputs back to values.
- **Error minimization**, achieved by optimizing both basis functions (tuning curves) and synaptic weights to reduce representation and transformation errors.

In short, basis functions allow NEF to encode and decode continuous variables, perform computations, and approximate dynamical behavior in a biologically plausible manner.

### 4.3 Why a Single Neuron Isn’t Enough

Through Nengo simulations with plotted visualizations, three cases demonstrate the limitations of using just one neuron for representation:

#### 4.3.1 High-dimensional input

Encoding a 2D vector $x=(0.5, 0.5)$ using a single neuron fails to reconstruct the signal. An ensemble of five neurons produces a significantly more accurate approximation, illustrating the need for multiple neurons to represent higher-dimensional vectors.

<p align="center"><img width="730" height="390" alt="image" src="https://github.com/user-attachments/assets/4fc1c93b-b6b9-41c7-8c32-298c43735875" /></p>

#### 4.3.2 Non-linear function $f(x)=x^2$

For a sinusoidal input in $[-1,1]$, a single neuron cannot approximate the quadratic transformation. A fifty-neuron ensemble, however, achieves a reasonable fit, showcasing the necessity of ensembles for non-linear mappings.

<p align="center"><img width="819" height="617" alt="image" src="https://github.com/user-attachments/assets/8949249b-0063-446d-be7e-3e1e33c83e63" />
</p>

#### 4.3.3 Accurate signal reproduction

Encoding a sinusoidal signal with one, two, and one hundred neurons reveals increasing fidelity as neuron count rises. A single neuron cannot accurately follow dynamic signals, whereas many neurons provide smoother, more precise reconstruction.

<p align="center"><img width="819" height="497" alt="image" src="https://github.com/user-attachments/assets/2431b98c-2844-48fa-b0a9-649cc61d2f9f" />
</p>

### 4.4 Radius: Its Role and Importance

The **radius** parameter defines the input scale that a neural ensemble can represent. Straying beyond this radius leads to inaccurate representation, regardless of neuron count.

**Example:** Two ensembles with 100 neurons each—one with radius 1, the other with radius 1.5 - receive input $(1,1)$. The radius-1 ensemble cannot decode this input accurately, while the radius-1.5 ensemble successfully represents it. Properly setting the radius ensures ensembles capture intended input ranges.

<p align="center"><img width="855" height="449" alt="image" src="https://github.com/user-attachments/assets/4671c738-667a-4a5f-94ed-4d38dcde0879" /></p>

### 4.5 Tuning Curves in Nengo

A **tuning curve** (I–F curve) maps input current $I$ to the neuron's firing rate $f$. Parameters like **encoder** (preferred stimulus direction) and **intercept** (minimum input to activate the neuron) define each neuron's response in Nengo.

Example: A neuron with $encoder=1$, $intercept=1$, only responds when $I \ge 1$.
<p align="center"><img width="371" height="279" alt="image" src="https://github.com/user-attachments/assets/cac50a27-1f65-425c-86f6-df4a9394cb1d" /></p>

Ensembles of 2 or 50 neurons produce distinct tuning curves, illustrating how mixed tuning patterns contribute to encoding and decoding performance. Larger ensembles with diverse tuning curves yield more accurate signal reconstruction.
<p align="center"><img width="857" height="226" alt="image" src="https://github.com/user-attachments/assets/e7d2daac-6f0f-4dd3-a9ff-9a9e3eb946c2" /></p>
<p align="center"><img width="862" height="227" alt="image" src="https://github.com/user-attachments/assets/fb9b4541-8f52-4092-b962-ebe2e0b0da27" /></p>


### 4.6 Conclusion

NEF, implemented via Nengo, provides a powerful high-level toolkit for building biologically grounded neural models. By leveraging ensembles, tuning curves, and structured connectivity, it can represent, compute, and simulate dynamics ranging from simple transforms to real oscillators and attractor networks. Careful tuning of parameters—such as neuron count, radius, and synaptic time constants—is key to balancing accuracy, realism, and efficiency.

</details>

## 5. Project Artifacts

<details>
<summary>Click to expand</summary>

### 5.1 Project Structure

```
neuromorphic-engineering/
├── morphologically_detailed_neuron_models/
├── point_neuronal_dynamic_models/
├── snn/
├── nengo/
└── README.md
```

### 5.2 Point Neuronal Dynamic Models

**Location**: `point_neuronal_dynamic_models/`

Focused on **single-compartment models** of neural spiking dynamics:

* `LIF.py`, `LIF-HW1.py`, `LIF-HW2.py`: Leaky Integrate-and-Fire neuron models.
* `Izhikevich.py`, `Izhikevich_ref.py`, `Izhikevich-HW.py`: Izhikevich neuron types with different firing regimes.
* `HnH.py`: Hodgkin–Huxley model implementation.
* `*_plots/`: Visualization of voltage traces and parameter variations for different models.

### 5.3 Morphologically Detailed Neuron Models

**Location**: `morphologically_detailed_neuron_models/`

This module contains implementations and simulations of morphologically accurate neuron models using:

* `the_compartmental_model.py`: Compartmental modeling of neurons.
* `the_cable_equation.py`: Classical cable equation for dendritic voltage propagation.
* `the_cable_equation_2.py`: A variant of the cable equation implementation.

### 5.4 Nengo Simulations

**Location**: `nengo/`

This submodule includes a wide variety of **simulations using the Nengo framework**, illustrating key concepts of neuromorphic computing and the Neural Engineering Framework (NEF).

#### 5.5 Highlights

##### 5.5.1 Representation

* `representation.py`: LIF vs. Rectified Linear tuning curves, ensemble decoding, high-dimensional analysis.

##### 5.5.2 Transformation

* `transformation.py`: Decoder-based function transformations (e.g., `sin(x)` to `sin²(x)`, vector sums, multiplication, gating).

##### 5.5.3 Dynamics

* `dynamics.py`: Recurrent connections modeling functions like `f(x)=x+1`, `f(x)=x²`, `f(x)=-x`, integrators, oscillators, and Lorenz attractor.

##### 5.5.4 Learning and Adaptation

* `pes.py`: Online learning using the PES rule in a simple communication channel.
* `pavlovian.py`: Classical conditioning via Hebbian and PES learning mechanisms.

##### 5.5.5 Interactive Experiments

**Location**: `nengo/HW/`

* Tuning curves (`tuning_curves.py`)
* Function approximations (`non_linear.py`, `transformation_hw.py`)
* Ring attractors (`ring_attractor.py`, `Ring_Attractor.ipynb`)
* High-dimensional stimuli (`high_dim_stim.py`)
* Performance metrics & visualizations (`accuracy.py`, `radius.py`)

## 5.6 Spiking Neural Networks (SNN)

**Location**: `snn/`

Advanced simulations of SNNs and learning algorithms.

* `diffrentiable_lif.py`: Tuning curves for LIF neurons with differentiable nonlinearities.
* `mnist.py`: A spiking convolutional neural network (SNN) for MNIST classification using **NengoDL**.
* `pavlovian.py`: Pavlovian conditioning using ensembles and learning rules.
* `pes.py`: Implementation of PES learning on a feedforward task.

</details>

## 6. Requirements

* Python ≥ 3.8
* `nengo`
* `nengo-dl`
* `tensorflow`
* `matplotlib`
* `numpy`
* `seaborn`
* `scipy`

## 7. References

- [Nengo](https://www.nengo.ai)
- [NengoDL](https://www.nengo.ai/nengo-dl)
- [Izhikevich, E. M. (2003). Simple model of spiking neurons](https://www.izhikevich.org/publications/spikes.pdf)
- [Photinus carolinus - Wikipedia](https://en.wikipedia.org/wiki/Photinus_carolinus)
- [Synchronous Fireflies - Firefly.org](https://www.firefly.org/synchronous-fireflies.html)
- [Synchronous Flashing of Fireflies (JSTOR - direct PDF)](https://www-jstor-org.elib.openu.ac.il/stable/pdf/2830425.pdf)
- [AI-Native Supercomputing and the Revival of Moore’s Law (Cambridge)](https://www.cambridge.org/core/journals/apsipa-transactions-on-signal-and-information-processing/article/ai-native-supercomputing-and-the-revival-of-moores-law/3791FFFAC8FCA71718FA360D0C8FC0D8?utm_campaign=shareaholic)
- [AI-Native Supercomputing (EBSCOhost login)](http://elib.openu.ac.il/login?url=https://search.ebscohost.com/login.aspx?direct=true&db=a9h&AN=154960632&site=eds-live&scope=site)
- [ArXiv: Delays and the Dynamics of Firefly Synchronization (PDF)](https://arxiv.org/pdf/1409.3215.pdf)
