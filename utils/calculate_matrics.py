from utils.util_f import *
import numpy as np
import pandas as pd

def calculate_precision():
    accuracy = 0.6359
    sensitivity = 0.9126  # Recall
    specificity = 0.5532
    # Assume total samples (N = 1 for simplicity in proportions)
    N = 1
    # Solve for TP, FN, TN, and FP using equations
    from sympy import symbols, Eq, solve
    # Define variables
    TP, TN, FN, FP = symbols('TP TN FN FP', positive=True)
    # Equations based on definitions
    eq1 = Eq(TP + TN, accuracy * N)  # Accuracy relationship
    eq2 = Eq(TP / (TP + FN), sensitivity)  # Sensitivity (Recall) definition
    eq3 = Eq(TN / (TN + FP), specificity)  # Specificity definition
    eq4 = Eq(TP + FN + TN + FP, N)  # Total samples = N
    # Solve the equations
    solution = solve((eq1, eq2, eq3, eq4), (TP, TN, FN, FP))
    # Extract values
    TP = solution[TP]
    TN = solution[TN]
    FN = solution[FN]
    FP = solution[FP]
    # Calculate precision
    precision = TP / (TP + FP)
    # Recall is already provided as sensitivity
    recall = sensitivity
    # Calculate F1 score
    f1_score = 2 * (precision * recall) / (precision + recall)
    # Print results
    print("True Positives (TP):", TP)
    print("True Negatives (TN):", TN)
    print("False Negatives (FN):", FN)
    print("False Positives (FP):", FP)
    print("Precision:", precision)
    print("F1 Score:", f1_score)


def calculate_accuracy(sensitivity, specificity, num_positives, num_negatives):
    # Calculate number of positives and negatives in the dataset
    # num_positives = prevalence * N
    # num_negatives = (1 - prevalence) * N

    # Calculate True Positives (TP), True Negatives (TN), False Positives (FP), and False Negatives (FN)
    TP = sensitivity * num_positives
    TN = specificity * num_negatives
    FP = (1 - specificity) * num_negatives
    FN = (1 - sensitivity) * num_positives

    # Calculate accuracy
    accuracy = (TP + TN) / (TP + TN + FP + FN)

    return accuracy
