"""
This module provides functions to calculate class weights
"""
def calc_class_weight(num_train, num_classes, class_counts):
    class_weight = {i: num_train / (num_classes * c) for i, c in enumerate(class_counts)}
    #np.log(max_cnt / cnt) + 1
    return class_weight
