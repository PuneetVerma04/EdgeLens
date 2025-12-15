import torch


def postprocess_output(prediction: torch.Tensor) -> dict:
    """
    Postprocess model output to human-readable format.
    
    Args:
        prediction: Raw model output tensor
        
    Returns:
        Dictionary with label and confidence
    """
    probabilities = torch.softmax(prediction, dim=1)
    confidence, class_idx = torch.max(probabilities, dim=1)

    label_map = {0: "Defective", 1: "OK"}

    return {
        "label": label_map[int(class_idx)],
        "confidence": float(confidence)
    }
