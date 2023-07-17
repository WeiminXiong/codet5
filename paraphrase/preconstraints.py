import nltk

from textattack.constraints import PreTransformationConstraint
from textattack.shared.validators import transformation_consists_of_word_swaps
from nltk import (pos_tag, word_tokenize)

noun = ["NN", "NNS", "NNP", "NNPS"]
adjective = ["JJ", "JJR", "JJS"]
adverb = ["RB", "RBR", "RBS"]
verb = ["VB", "VBD", "VBG", "VBN", "VBP", "VBZ"]

class ProgramConstraint(PreTransformationConstraint):
    def __init__(self) -> None:
        super().__init__()
        self.program_constraint = ["def", "list", "set", "function", "parameter", "return",
                                   "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
                                   "uranus", "lst", "arr", "strongestextensionname", "sm", "num",
                                   "first", "second", "third", "fourth", "fifth", "sixth", "seventh", "eighth"]
        
        
    def _get_modifiable_indices(self, current_text):
        """Returns the word indices in ``current_text`` which are able to be
        modified."""
        editable_word_indices = set()
        for i, word in enumerate(current_text.words):
            if word.lower() not in self.program_constraint:
                if len(word) > 1 and word.isalpha():
                    editable_word_indices.add(i)
        
        return editable_word_indices
    

class PosPreConstraint(PreTransformationConstraint):
    def __init__(self) -> None:
        super().__init__()
        
        
    def _get_modifiable_indices(self, current_text):
        tokens = current_text.words
        editable_word_indices = set()
        pos_tags = pos_tag(tokens)
        for i, tag in enumerate(pos_tags):
            if tag[1] in noun or tag[1] in adjective or tag[1] in adverb or tag[1] in verb:
                editable_word_indices.add(i)
        
        return editable_word_indices
        