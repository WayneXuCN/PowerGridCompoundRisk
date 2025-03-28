import numpy as np

def calculate_failure_rate(states, threshold=0.2):
    """计算失效节点比例"""
    return np.mean(states < threshold, axis=1)

def find_critical_nodes(failure_probs, final_states, prob_thresh=0.7, state_thresh=0.2):
    """识别关键节点"""
    return np.where(
        (failure_probs > prob_thresh) & (final_states < state_thresh)
    )[0]