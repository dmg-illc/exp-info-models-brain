import numpy as np
import re

def get_relevant_tokens_indices(offset_mapping, full_prompt, target_word):
  
  """
    Offset mapping should be a numpy array with the offset mapping of a single sentence. 
    Full prompt is the entire sentence, e.g. 'This is a corkscrew.'
    Target word is the word that should be found within the sentence, e.g., 'corkscrek'.

    The function returns the indices of the offset mappings that correspond to the 
    target words (so the indices that can be used for the indexing the hidden states).
  """

  start_ind = full_prompt.find(target_word)
  end_ind = start_ind + len(target_word)
  print(start_ind, end_ind)
  ret_ind = []
  for i in range(offset_mapping.shape[0]):
    if (offset_mapping[i,0] >= start_ind) & (offset_mapping[i,1]<= end_ind):
      ret_ind.append(i)
  return np.array(ret_ind)




def add_indefinite_article(template: str, word: str) -> str:
    vowels = "aeiou"
    # Special case for silent 'h' and some exceptions
    exceptions = {"hour", "honor", "heir", "herb"}  # 'herb' is only in American English
    
    if word.lower() in exceptions or (word[0].lower() in vowels and word.lower() != "user"):
        article = "an"
    else:
        article = "a"
    
    return re.sub(r'\{article\}', article, template).replace("{word}", word).capitalize()


def get_input_sentences(current_templates, word):
    """
      Given a list of templates and a word, outputs a list of templates filled in with 
      the target word and, if necessary, the proper indefinite article.
    """

    input_templates = []
    for template in current_templates:
        if '{article}' in template:
            input_templates.append(add_indefinite_article(template, word))
        else:
            input_templates.append(template.format(word=word))
    return input_templates