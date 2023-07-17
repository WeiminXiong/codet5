from textattack.augmentation import Augmenter
from textattack.attack import AttackedText

class myAugmenter(Augmenter):
    def __init__(self, transformation, constraints=..., pct_words_to_swap=0.1, transformations_per_example=1, high_yield=False, fast_augment=False, enable_advanced_metrics=False):
        super().__init__(transformation, constraints, pct_words_to_swap, transformations_per_example, high_yield, fast_augment, enable_advanced_metrics)
    
    def augment(self, text):
        attacked_text = AttackedText(text)
        original_text = attacked_text
        transformed_texts = self.transformation(
                    attacked_text, self.pre_transformation_constraints
                )

        # Filter out transformations that don't match the constraints.
        transformed_texts = self._filter_transformations(
            transformed_texts, attacked_text, original_text
        )
        return [item.text for item in transformed_texts]