"""
Classification module for sentiment analysis and topic modeling
"""
from .classifier import EVOpinionClassifier
from .evaluation import EVClassifierEvaluator, create_annotation_tool

__all__ = ['EVOpinionClassifier', 'EVClassifierEvaluator', 'create_annotation_tool']