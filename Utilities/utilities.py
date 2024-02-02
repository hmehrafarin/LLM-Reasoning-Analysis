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

def generate_random_words(n):
    words = []
    for _ in range(n):
        word_length = random.randint(1, 8)  # you can set your own limits
        word = ''.join(random.choice(string.ascii_lowercase) for _ in range(word_length))
        words.append(word)
    return words

def match_answer_with_facts(fact1, fact2, answers):
    answers = re.sub(r'\(.*?\)', '', answers)
    answers = answers[1:]
    answers_list = answers.split("  ")
    input_string1 = fact1.translate(str.maketrans('', '', string.punctuation.replace('-', '')))
    input_string2 = fact2.translate(str.maketrans('', '', string.punctuation.replace('-', '')))
    input_string1, input_string2 = input_string1.lower(), input_string2.lower()
    for token2 in answers_list:
        token2 = token2.lower()
        if token2.lower() + ' ' in input_string1.lower():
            input_string1 = input_string1.replace(token2 + ' ', '')
        elif token2 in input_string1:
            input_string1 = input_string1.replace(token2, '')
        if token2 + ' ' in input_string2:
            input_string2 = input_string2.replace(token2 + ' ', '')
        elif token2 in input_string2:
            input_string2 = input_string2.replace(token2, '')
    return input_string1, input_string2


def ablate_connecting_words(input_string1, input_string2, out_dict):
    stemmer = PorterStemmer()

    input_string1, input_string2 = input_string1.translate(str.maketrans('', '', string.punctuation)), \
                                   input_string2.translate(str.maketrans('', '', string.punctuation))
    listed_string1, listed_string2 = input_string1.split(" "), input_string2.split(" ")
    listed_string1_copy, listed_string2_copy = listed_string1[:], listed_string2[:]
    
    list_of_ablted_words = []
    for token in listed_string1_copy:
        for token2 in listed_string2_copy:
            if stemmer.stem(token.lower()) == stemmer.stem(token2.lower()):
                if token in listed_string1:
                        listed_string1.remove(token)
                        list_of_ablted_words.append(token) if token not in list_of_ablted_words else None
                if token2 in listed_string2:
                        listed_string2.remove(token2)

    input_string1, input_string2 = " ".join(listed_string1), " ".join(listed_string2)
    input_string1 = " " if input_string1 == "" else input_string1
    input_string2 = " " if input_string2 == "" else input_string2
    if list_of_ablted_words == []:
        list_of_ablted_words.append("None")
    out_dict['ablated tokens'].append(list_of_ablted_words)

    return input_string1, input_string2

def random_token_ablator(input_string, num_tokens):
    listed_string = input_string.split(" ")
    for i in range(num_tokens):
        random_generated_number = random.randrange(len(listed_string))
        while listed_string[random_generated_number] in stopwords.words('english'):
            random_generated_number = random.randrange(len(listed_string))
        listed_string.pop(random_generated_number)
    input_string = " ".join(listed_string) + "."
    return input_string

def string_shuffler(input_string):
    listed_string = input_string.split(" ")
    input_string = " ".join(random.sample(listed_string, len(listed_string))) + "."
    return input_string
