# Real Time Object Detection
The following repository is a collection of research & production grade scripts allowing you to deploy a real-time object detection inference client using various methods.
The main goal was to research methods on deploying this kind of system at large scale, leveraging hardware to its limit, while building lightweight and fast software.

## Triton Server Client
The [`client-triton`](client-triton) includes a client implementation based on **NVIDIA's Triton Server**.<br>
The entire processing is done in a dedicated client written in **Rust**, to minimize overhead when doing inference at large scale.<br>
The following architecture takes place:<br>

<img src="client-triton/assets/architecture.png" alt="Architecture" width="700"/>

## DeepStream Client
The [`client-deepstream`](client-deepstream) includes a POC implementation based on **NVIDIA's DeepStream**.<br>
Due to DeepStream's approach of being more of a complete system for real time inference, and the fact that it requires us to write code that is mainly based around the DeepSeek SDK, made be abandon this path, with the idea that a more general purpose and flexible system is suitable for this task.