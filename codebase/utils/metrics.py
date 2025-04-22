import torch

def feature_to_segments(feature_tensor: torch.Tensor) -> torch.Tensor:
    """
    Given a feature tensor of shape [B, T, F], this function computes 
    the per-sample segmentation based on the location of the largest change.
    It returns a tensor of shape [B, 2, 2] where for each sample the segments are:
      [[0, boundary], [boundary, T]]
    """
    B, T, F = feature_tensor.shape
    # Compute differences along the temporal axis (T)
    diffs = torch.diff(feature_tensor, dim=1)         # shape: [B, T-1, F]
    diff_norm = torch.norm(diffs.float(), dim=-1)       # shape: [B, T-1]
    # Find the index where the norm is maximized for each sample
    boundary_indices = torch.argmax(diff_norm, dim=1, keepdim=True) + 1  # shape: [B, 1]
    
    # Create the first segment: from 0 to boundary_indices
    first_segments = torch.cat((torch.zeros_like(boundary_indices), boundary_indices), dim=1)  # shape: [B, 2]
    # Create the second segment: from boundary_indices to T
    T_tensor = torch.full_like(boundary_indices, T)
    second_segments = torch.cat((boundary_indices, T_tensor), dim=1)  # shape: [B, 2]
    
    # Stack the segments so that each sample returns a [2, 2] tensor.
    segments = torch.stack((first_segments, second_segments), dim=1)  # shape: [B, 2, 2]
    return segments

def compute_iou_matrix(pred_segments: torch.Tensor, true_segments: torch.Tensor) -> torch.Tensor:
    """
    Compute the IoU for segments.
    Both inputs are tensors of shape [B, N, 2] and [B, M, 2], where the last dimension represents (start, end).
    Returns a tensor of shape [B, N, M] containing IoU values for each pair.
    """
    # Separate start and end indices for predicted and true segments
    pred_start = pred_segments[..., 0]  # [B, N]
    pred_end   = pred_segments[..., 1]  # [B, N]
    true_start = true_segments[..., 0]  # [B, M]
    true_end   = true_segments[..., 1]  # [B, M]
    
    # Compute the pairwise maximum of start indices and minimum of end indices via broadcasting
    start_max = torch.max(pred_start.unsqueeze(2), true_start.unsqueeze(1))  # [B, N, M]
    end_min   = torch.min(pred_end.unsqueeze(2), true_end.unsqueeze(1))      # [B, N, M]
    
    # Compute intersection length and union length
    intersection = (end_min - start_max).clamp(min=0)
    union = torch.max(pred_end.unsqueeze(2), true_end.unsqueeze(1)) - torch.min(pred_start.unsqueeze(2), true_start.unsqueeze(1))
    
    # Compute IoU, using a safe division to avoid dividing by zero.
    iou = torch.where(union > 0, intersection / union, torch.zeros_like(intersection))
    return iou

def segmentation_covering_f1(pred_feature: torch.Tensor,
                                           true_feature: torch.Tensor,
                                           iou_thresh: float = 0.5):
    """
    Computes the segmentation covering F1 metric between predicted and true features
    and includes an accuracy metric computed from the IoU matches.
    
    Both pred_feature and true_feature are expected to be tensors of shape [B, T, F].
    
    The function does the following:
      1. Computes segments for both prediction and ground truth.
      2. Computes the pairwise IoU matrix for these segments.
      3. Determines matches by comparing the IoU values to a threshold.
      4. Computes precision and recall as the proportion of predicted and true segments,
         respectively, that have at least one matching segment.
      5. Computes the F1 score from precision and recall.
      6. Computes accuracy as the fraction of samples where both predicted segments 
         (and correspondingly, both ground truth segments) have a match.
    
    Returns:
      avg_precision, avg_recall, avg_f1, avg_accuracy  (all scalar tensors)
    """
    # Compute segments for both prediction and ground truth (shape: [B, 2, 2])
    pred_segments = feature_to_segments(pred_feature)
    true_segments = feature_to_segments(true_feature)
    
    # Compute IoU matrix for each sample (shape: [B, 2, 2])
    iou_matrix = compute_iou_matrix(pred_segments, true_segments)
    
    # Determine matches based on the IoU threshold.
    # A match is True if the IoU for a pair of segments exceeds the threshold.
    matches = iou_matrix >= iou_thresh

    # For each predicted segment, check if any ground truth segment has sufficient IoU.
    pred_match = matches.any(dim=2).float()  # shape: [B, 2]
    # For each true segment, check if any predicted segment has sufficient IoU.
    true_match = matches.any(dim=1).float()  # shape: [B, 2]
    
    # Count true positives per sample.
    tp_pred = pred_match.sum(dim=1)  # number of matched predicted segments per sample
    tp_true = true_match.sum(dim=1)  # number of matched ground truth segments per sample
    
    # Number of segments in each set (expected to be 2).
    num_pred = pred_segments.shape[1]
    num_true = true_segments.shape[1]
    
    # Compute precision and recall per sample.
    precision = tp_pred / num_pred
    recall = tp_true / num_true
    
    # Compute F1 score with safe division.
    f1 = torch.where((precision + recall) > 0,
                     2 * precision * recall / (precision + recall),
                     torch.zeros_like(precision))
    
    # Average the metrics over the batch.
    avg_precision = precision.mean()
    avg_recall = recall.mean()
    avg_f1 = f1.mean()
    
    # Compute segmentation accuracy using IoU matches.
    # Here we define a sample as correct only if all predicted segments (and thus all true segments)
    # have a matching segment based on the IoU threshold.
    sample_accuracy = ((pred_match.sum(dim=1) == num_pred) & 
                       (true_match.sum(dim=1) == num_true)).float()
    avg_accuracy = sample_accuracy.mean()
    
    return avg_precision, avg_recall, avg_f1, avg_accuracy
