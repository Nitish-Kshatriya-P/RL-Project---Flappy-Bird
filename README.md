# 🐦 Flappy Bird AI – Deep Reinforcement Learning Agent

## 📌 Overview

This project implements a Deep Q-Network (DQN) agent that learns to play Flappy Bird autonomously.
Using reinforcement learning techniques like experience replay and target networks, the agent improves over time by trial and error, eventually mastering the game.

## ⚙️ Features

    🎮 Autonomous Gameplay – AI agent learns to play Flappy Bird without human intervention.
    
    🧠 Deep Q-Network (DQN) – Neural network approximates Q-values for decision-making.
    
    🔄 Experience Replay – Past experiences are stored and replayed to stabilize training.
    
    🎯 Target Network Updates – Improves training stability and prevents divergence.
    
    💾 Model Checkpointing – Saves best-performing models and logs training progress.
    
    🖥️ Training & Testing Modes – Train from scratch or test pre-trained models with rendering.

## 🏗️ Architecture

      Environment (Flappy Bird Gym)
                 ↓
         Neural Network (DQN)
                 ↓
       Action Selection (ε-greedy)
                 ↓
       Experience Replay Memory
                 ↓
      Q-value Update + Target Network

## 🚀 Getting Started
    1️⃣ Clone the Repository
    git clone <repo-link>
    cd flappy-bird-rl
    
    2️⃣ Install Dependencies
    pip install -r requirements.txt
    
    3️⃣ Train the Agent
    python main.py
    
    4️⃣ Test the Agent (with rendering)
    Switch to the test() function in main function.

## 🧑‍💻 Tech Stack

    Python 3.9+
    
    PyTorch – Deep Q-Network implementation
    
    Gymnasium + Flappy Bird Environment – RL environment for training/testing
    
    YAML – Hyperparameter configuration

## 📊 Training Workflow

    Agent starts with random actions (exploration via ε-greedy).
    
    Rewards are collected and stored in ReplayMemory.
    
    DQN learns from batches of past experiences.
    
    Target network updated periodically for stability.
    
    Model checkpoints saved when performance improves.
