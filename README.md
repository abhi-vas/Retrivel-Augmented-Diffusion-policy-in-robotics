# Retrieval-Augmented Diffusion Policy for Robotics

## Overview

**Retrieval-Augmented Diffusion Policy (RA-Diffusion Policy)** combines **retrieval-augmented learning** with **diffusion-based visuomotor control** for robotic manipulation.

The main idea is to provide the policy with relevant past robotic experiences retrieved from a demonstration dataset. Instead of generating actions only from the current observation, the policy retrieves similar trajectories and uses them as additional context for action generation.

```text
           Current Observation
                  ↓
             Multimodal Encoder
                  ↓
              Retrieval
                  ↓
        Relevant Demonstrations
                  ↓
          Diffusion Policy
                  ↓
            Action Sequence
                  ↓
                Robot
```

## Motivation

A conventional Diffusion Policy learns a mapping from observations to actions:

```text
Observation → Diffusion Policy → Action Sequence
```

While this approach can learn complex manipulation behaviors, the policy primarily relies on information encoded in its learned parameters.

Retrieval augmentation introduces an **external memory** containing previously observed robotic demonstrations:

```text
   Current  Observation
          ↓
       Retriever
          ↓
Relevant Past Experiences
          ↓
   Diffusion Transformer
          ↓
    Action Trajectory
```

This allows the policy to explicitly access demonstrations that are similar to the current task.

## Key Components

### 1. Multimodal Encoder

The system converts robotic observations and task instructions into vector representations.

Depending on the available data, the encoder can process:

* RGB images
* Video observations
* Natural-language instructions
* Robot states
* Other modalities

These embeddings are used to measure similarity between the current task and previously collected demonstrations.

### 2. Retrieval System

A vector index is constructed from previously collected robot trajectories.

Each demonstration is converted into an embedding and stored in a vector database using an approximate nearest-neighbor search method such as **HNSW** or **FAISS**.

At inference time:

```text
Current Instruction + Observation
                ↓
         Query Embedding
                ↓
        Similarity Search
                ↓
          Top-K Retrievals
                ↓
       Relevant Trajectories
```

### 3. Retrieved Demonstrations

Each stored demonstration can contain:

```text
Instruction
Observation Sequence
Robot State
Action Sequence
```

For example:

```text
Trajectory =
{
    states,
    actions
}
```

The retrieved trajectories provide additional contextual information to the policy.

### 4. Diffusion Policy

The retrieved context is combined with the current observation and task instruction and provided to a diffusion-based policy.

The diffusion model generates a sequence of actions rather than predicting a single action independently.

```text
Current Observation
        +
Retrieved Demonstrations
        ↓
 Diffusion Transformer
        ↓
  Diffusion Process
        ↓
 Action Trajectory
```

## Retrieval Pipeline

The complete retrieval pipeline consists of the following steps:

1. Collect robot demonstration trajectories.
2. Extract observations, states, and actions.
3. Generate multimodal embeddings.
4. Store embeddings in a vector index.
5. Receive a new task instruction and observation.
6. Generate an embedding for the query.
7. Retrieve the most similar demonstrations.
8. Construct the retrieval-augmented context.
9. Pass the context to the diffusion policy.
10. Generate the robot action trajectory.

```text
                 ┌──────────────────────┐
                 │ Robot Demonstrations  │
                 └──────────┬───────────┘
                            ↓
                  Multimodal Encoder
                            ↓
                       Embeddings
                            ↓
                    Vector Database
                            │
                            │
New Task + Observation ────┘
                            ↓
                       Retrieval
                            ↓
                 Top-K Demonstrations
                            ↓
                  Diffusion Transformer
                            ↓
                    Action Sequence
                            ↓
                          Robot
```

## Trajectory Representation

A demonstration is represented as a sequence of observations and actions:

```text
Instruction:
"Pick up the object."

Observations:
O₁ → O₂ → O₃ → ... → Oₜ

Robot States:
S₁ → S₂ → S₃ → ... → Sₜ

Actions:
A₁ → A₂ → A₃ → ... → Aₜ
```

The retrieval system searches the demonstration database for trajectories that are visually, semantically, or contextually similar to the current task.

## Example

Suppose the robot receives:

```text
"Pick up the red cube."
```

The current observation is converted into an embedding and compared with the stored demonstrations.

The retriever may return:

```text
1. Pick up a red cube
2. Pick up a blue cube
3. Move an object to another location
```

These retrieved demonstrations are then provided to the diffusion policy:

```text
Current Observation
        +
Task Instruction
        +
Retrieved Demonstrations
        ↓
Diffusion Transformer
        ↓
Predicted Action Sequence
```

The robot executes the generated action trajectory.

## Advantages

### External Episodic Memory

The retrieval database provides an external memory of previously observed robotic experiences.

### Better Use of Demonstrations

Instead of relying only on learned model parameters, the policy can explicitly access relevant demonstrations during inference.

### Improved Generalization

Similar demonstrations may help the policy adapt to variations in objects, scenes, and task configurations.

### Multimodal Retrieval

The retrieval system can incorporate information from multiple modalities, including:

* Vision
* Language
* Video
* Robot state

### Scalable Memory

New demonstrations can be added to the retrieval database without necessarily retraining the entire policy.

## System Architecture

```text
                         ┌─────────────────────┐
                         │ Demonstration Data  │
                         └──────────┬──────────┘
                                    ↓
                         ┌─────────────────────┐
                         │ Multimodal Encoder  │
                         └──────────┬──────────┘
                                    ↓
                         ┌─────────────────────┐
                         │  Vector Database    │
                         │    HNSW / FAISS     │
                         └──────────┬──────────┘
                                    │
                                    │ Retrieve
                                    ↓
┌─────────────────┐       ┌─────────────────────┐
│ Current Task    │──────→│ Retrieval Module    │
│ + Observation   │       └──────────┬──────────┘
└─────────────────┘                  ↓
                           ┌─────────────────────┐
                           │ Retrieved Context  │
                           └──────────┬──────────┘
                                      ↓
                           ┌─────────────────────┐
                           │ Diffusion Policy    │
                           │    Transformer      │
                           └──────────┬──────────┘
                                      ↓
                           ┌─────────────────────┐
                           │   Action Sequence   │
                           └──────────┬──────────┘
                                      ↓
                                   Robot
```

## Project Structure

```text
retrieval-augmented-diffusion-policy/
│
├── data/
│   ├── dataset.py
│   └── trajectory_processing.py
│
├── retrieval/
│   ├── embedding.py
│   ├── index.py
│   └── retriever.py
│
├── encoder/
│   └── multimodal_encoder.py
│
├── diffusion_policy/
│   ├── model.py
│   ├── transformer.py
│   └── diffusion.py
│
├── configs/
│   └── config.yaml
│
├── train.py
├── inference.py
└── README.md
```

## Research Objective

The objective of this project is to investigate whether **retrieval-augmented contextual information can improve the performance and generalization of diffusion-based robotic policies**.

The project focuses on evaluating whether retrieved demonstrations can help the robot:

* Improve manipulation success rate
* Generalize to unseen task configurations
* Make better use of large demonstration datasets
* Leverage similar previous experiences
* Improve action generation through additional contextual information

## Overall Concept

The conventional approach is:

```text
Observation → Policy → Action
```

The proposed retrieval-augmented approach is:

```text
                 ┌──────────────┐
                 │  Instruction │
                 └──────┬───────┘
                        ↓
                 ┌──────────────┐
                 │ Observation  │
                 └──────┬───────┘
                        ↓
                 ┌──────────────┐
                 │  Retrieval   │
                 └──────┬───────┘
                        ↓
             ┌─────────────────────┐
             │ Relevant Experiences│
             └──────────┬──────────┘
                        ↓
             ┌─────────────────────┐
             │  Diffusion Policy   │
             └──────────┬──────────┘
                        ↓
                 ┌──────────────┐
                 │Action Sequence│
                 └──────────────┘
```

The resulting system combines **retrieval-based external memory** with **diffusion-based action generation**, providing a framework for retrieval-augmented robotic manipulation.
