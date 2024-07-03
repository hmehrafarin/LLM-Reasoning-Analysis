import numpy as np
import torch
import random
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import string
import re

def custom_set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def generate_random_words(input):
    # Get lengths of fact 1 and fact 2
    fact1_length, fact2_length = len(input['fact 1'].split(" ")), len(input['fact 2'].split(" "))
    words = []
    
    # Generate x raondom words of length between 2 and 8 (x = fact1_length or fact2_length)
    for _ in range(fact1_length):
        word_length = random.randint(2, 8)  # you can set your own limits
        word = ''.join(random.choice(string.ascii_lowercase) for _ in range(word_length))
        words.append(word)
    input['fact 1'] = " ".join(words)
    words = []
    for _ in range(fact2_length):
        word_length = random.randint(2, 8)  # you can set your own limits
        word = ''.join(random.choice(string.ascii_lowercase) for _ in range(word_length))
        words.append(word)
    input['fact 2'] = " ".join(words)

    return input

def match_answer_with_facts(input):
    fact1, fact2, answers = input['fact 1'], input['fact 2'], input['answers']  

    # Extract choice texts from answers
    answers_list = re.sub(r'\(.*?\)', '', answers)[1:].split("  ")  

    # Remove punctuation and convert to lowercase
    fact1 = fact1.translate(str.maketrans('', '', string.punctuation.replace('-', ''))).lower()
    fact2 = fact2.translate(str.maketrans('', '', string.punctuation.replace('-', ''))).lower() 

    # Remove keywords from facts
    for token in answers_list:
        token = token.lower()
        for fact_key in ['fact 1', 'fact 2']:
            fact = input[fact_key]
            fact = fact.replace(token + ' ', '').replace(token, '')
            input[fact_key] = fact

    return input

def ablate_connecting_words(input, action ,out_dict):
    stemmer = PorterStemmer()
    strings = [input[action[i]].translate(str.maketrans('', '', string.punctuation)).split(" ") for i in range(2)]
    
    ablated_words = set(strings[0]) & set(strings[1])
    stemmed_ablated_words = {word for word in ablated_words if stemmer.stem(word.lower()) in {stemmer.stem(token.lower()) for token in ablated_words}}
    
    for i in range(2):
        if action[i] == 'question':
            continue
        strings[i] = [word for word in strings[i] if word not in stemmed_ablated_words]
        input[action[i]] = " ".join(strings[i]) or " "
    
    out_dict['ablated tokens'].append(list(stemmed_ablated_words) or ["None"])

    return input

def string_shuffler(input, entry):
    input_string = input[entry]
    if input_string.endswith("."):
        input_string = input_string[:-1]
    listed_string = input_string.split(" ")
    input_string = " ".join(random.sample(listed_string, len(listed_string)))
    input[entry] = input_string
    return input

def random_token_ablator(input, entry, num_tokens):
    input_string = input[entry]
    listed_string = input_string.split(" ")
    for i in range(num_tokens):
        random_generated_number = random.randrange(len(listed_string))
        while listed_string[random_generated_number] in stopwords.words('english'):
            random_generated_number = random.randrange(len(listed_string))
        listed_string.pop(random_generated_number)
    input_string = " ".join(listed_string) + "."
    input[entry] = input_string
    return input