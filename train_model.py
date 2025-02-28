# Копируем содержимое существующего файла train_model.py
import torch
import torch.nn as nn
import argparse
import os
import json
from model import IterativeReasoningModel, ReasoningDataset, train_model, evaluate_model

# ... существующий код ...
