class DecisionTreeException(Exception):
    """
    Base exception for all errors specific to the DecisionTree class.
    It extends the standard Python 'Exception' class.
    """
    pass

class NotTrainedError(DecisionTreeException):
    """
    Raised when prediction is attempted on a tree that has not been trained.
    Replaces the standard 'RuntimeError' for clarity.
    """
    pass

class InvalidInputError(DecisionTreeException):
    """
    Raised when input data (X or Y) is empty or has mismatched row counts.
    Replaces the standard 'ValueError' for clarity.
    """
    pass

class CriteriumLibraryError(DecisionTreeException):
    """Base exception for any error originating from the external Criterium or Metric library."""
    pass

class GainCalculationError(CriteriumLibraryError):
    """Raised specifically when the external Criterium.gain method fails."""
    pass