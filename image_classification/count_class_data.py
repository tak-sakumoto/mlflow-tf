"""
This module provides functions to count data in each class
"""
import numpy as np

def count_class_data(ds, num_classes):
    class_counts = np.array([0 for _ in range(num_classes)])
    for x, y in ds:
        class_counts[y.numpy()] += 1
    
    return class_counts