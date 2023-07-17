from textattack.transformations import Transformation
import json
from nltk import pos_tag
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch


class WordTransformation(Transformation):
    def __init__(self, dict = None, map_path = None):
        if map_path!=None:
            self.map_candidates = json.load(open(map_path, 'r'))
        if dict!=None:
            self.map_candidates = dict
        
    def recover_word_case(self, word, reference_word):
        """Makes the case of `word` like the case of `reference_word`.

        Supports lowercase, UPPERCASE, and Capitalized.
        """
        if reference_word.islower():
            return word.lower()
        elif reference_word.isupper() and len(reference_word) > 1:
            return word.upper()
        elif reference_word[0].isupper() and reference_word[1:].islower():
            return word.capitalize()
        else:
            # if other, just do not alter the word's case
            return word    
    
    def _get_replacement_words(self, word, tag):
        try:
            return self.map_candidates[f"{word}_{tag}"]
        except Exception:
            return []
    
    def extra_repr_keys(self):
        return super().extra_repr_keys() + ['map_candidates']
    
    def _get_transformations(self, current_text, indices_to_modify):
        words = current_text.words
        transformed_texts = []
        transformed_text = current_text
        pos_tags = pos_tag(words)
        for i in indices_to_modify:
            word_to_replace = words[i]
            
            replacement_words = self._get_replacement_words(word_to_replace.lower(), pos_tags[i][1])
            for r in replacement_words:
                if r==word_to_replace.lower():
                    continue
                if r in word_to_replace.lower() or word_to_replace.lower() in r:
                    continue
                t = self.recover_word_case(word=r, reference_word=word_to_replace)
                transformed_text = transformed_text.replace_word_at_index(i, t)
        
        transformed_texts.append(transformed_text)
        return transformed_texts


class WordTransformationWithLogits(Transformation):
    def __init__(self, dict = None, map_path = None):
        if map_path!=None:
            self.map_candidates = json.load(open(map_path, 'r'))
        if dict!=None:
            self.map_candidates = dict
        self.device = "cuda:0"
        self.model = AutoModelForSeq2SeqLM.from_pretrained('Salesforce/codet5p-770m-py').to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained('Salesforce/codet5p-770m-py')
        self.output_num = 10
        
    def recover_word_case(self, word, reference_word):
        """Makes the case of `word` like the case of `reference_word`.

        Supports lowercase, UPPERCASE, and Capitalized.
        """
        if reference_word.islower():
            return word.lower()
        elif reference_word.isupper() and len(reference_word) > 1:
            return word.upper()
        elif reference_word[0].isupper() and reference_word[1:].islower():
            return word.capitalize()
        else:
            # if other, just do not alter the word's case
            return word    
    
    def _get_replacement_words(self, word, tag):
        try:
            return self.map_candidates[f"{word}_{tag}"]
        except Exception:
            return []
    
    def extra_repr_keys(self):
        return super().extra_repr_keys() + ['map_candidates']
    
    def _get_list_transformations_at_index(self, current_texts, index):
        candidate_texts = []
        for text in current_texts:
            candidate_texts.extend(self._get_transformation_at_index(text, index))
        probabilities = self._calculate_probability([t.text for t in candidate_texts])
        _, sorted_indices = torch.sort(probabilities, descending=True)
        if sorted_indices.shape[0]>self.output_num:
            sorted_indices = sorted_indices[:self.output_num]
        return [candidate_texts[i] for i in sorted_indices]
            
    def _get_transformation_at_index(self, current_text, index):
        words = current_text.words
        transformed_texts = []
        transformed_texts.append(current_text)
        pos_tags = pos_tag(words)
        
        word_to_replace = words[index]
        replacement_words = self._get_replacement_words(word_to_replace.lower(), pos_tags[index][1])
        for r in replacement_words:
            if r==word_to_replace.lower():
                continue
            if r in word_to_replace.lower() or word_to_replace.lower() in r:
                continue
            t = self.recover_word_case(word=r, reference_word=word_to_replace)
            transformed_text = current_text.replace_word_at_index(index, t)
            transformed_texts.append(transformed_text)
        return list(set(transformed_texts))
    
    def _get_transformations(self, current_text, indices_to_modify):
        current_texts = [current_text]
        for i in indices_to_modify:
            current_texts = self._get_list_transformations_at_index(current_texts, i)
        return current_texts
            
                
    def _calculate_probability(self, augmented_texts):
        batch_size = len(augmented_texts)
        start_tokens = ['<s>']*batch_size
        encodings = self.tokenizer(start_tokens, return_tensors='pt')
        encodings = {k: v.to(self.device) for k, v in encodings.items()}
        labels = self.tokenizer(augmented_texts, return_tensors='pt', padding=True)['input_ids'].to(self.device)
        outputs = self.model(**encodings, labels=labels)
        logits = outputs['logits'].detach()
        logits_softmax = torch.softmax(logits, dim=-1)
        
        labels_token_prob_list = [logits_softmax[i, range(labels.shape[-1]), labels[i, :]] for i in range(batch_size)]
        labels_token_prob_list = torch.stack(labels_token_prob_list)
        labels_token_prob_list[labels==0]=1
        labels_token_prob_list = torch.log(labels_token_prob_list)
        labels_token_prob_list = torch.sum(labels_token_prob_list, dim=-1)/torch.sum(labels!=0, dim=-1)
        
        return labels_token_prob_list