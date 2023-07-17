from textattack.constraints import Constraint
from nltk import word_tokenize
from nltk import pos_tag


class PosConstraint(Constraint):
    def __init__(self, compare_against_original=True):
        super().__init__(compare_against_original)
    
    def _check_constraint(self, transformed_text, reference_text):
        reference_words = word_tokenize(reference_text.text)
        transformed_words = word_tokenize(transformed_text.text)
        
        reference_pos = pos_tag(reference_words)
        transformed_pos = pos_tag(transformed_words)
        
        reference_pos_tags = [item[1] for item in reference_pos]
        transformed_pos_tags = [item[1] for item in transformed_pos]
        
        return reference_pos_tags == transformed_pos_tags