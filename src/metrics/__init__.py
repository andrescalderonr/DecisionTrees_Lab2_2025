from .Accuracy import Accuracy
from .Precision import Precision
from .Recall import Recall
from .F1Score import F1Score

Metric.metric_map = {
    "accuracy": Accuracy,
    "precision": Precision,
    "recall": Recall,
    "f1score": F1Score
}
