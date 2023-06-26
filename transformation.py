from textattack.augmentation import EmbeddingAugmenter


augmenter = EmbeddingAugmenter(transformations_per_example=10)
s = "Check if in given list of numbers, are any two numbers closer to each other than\n    given threshold.\n    "
# s= 'What I cannot create, I do not understand.'
result = augmenter.augment(s)
for item in result:
    print(item)