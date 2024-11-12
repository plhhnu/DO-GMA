import os
import random
import numpy as np
import torch
import dgl
import logging

CHARPROTSET = {
    "A": 1,
    "C": 2,
    "B": 3,
    "E": 4,
    "D": 5,
    "G": 6,
    "F": 7,
    "I": 8,
    "H": 9,
    "K": 10,
    "M": 11,
    "L": 12,
    "O": 13,
    "N": 14,
    "Q": 15,
    "P": 16,
    "S": 17,
    "R": 18,
    "U": 19,
    "T": 20,
    "W": 21,
    "V": 22,
    "Y": 23,
    "X": 24,
    "Z": 25,
}

CHARPROTLEN = 25


def set_seed(seed=1000):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def graph_collate_func(x):
    d, p, y = zip(*x)
    d = dgl.batch(d)
    return d, torch.tensor(np.array(p)), torch.tensor(y)

def graph_collate_func_2(x):
    d,d_2, p, y = zip(*x)
    d = dgl.batch(d)
    return d,torch.tensor(np.array(d_2)), torch.tensor(np.array(p)), torch.tensor(y)


def mkdir(path):
    path = path.strip()
    path = path.rstrip("\\")
    is_exists = os.path.exists(path)
    if not is_exists:
        os.makedirs(path)
CHARISOSMISET = {
    '(': 1,
    '.': 2,
    '0': 3,
    '2': 4,
    '4': 5,
    '6': 6,
    '8': 7,
    '@': 8,
    'B': 9,
    'D': 10,
    'F': 11,
    'H': 12,
    'L': 13,
    'N': 14,
    'P': 15,
    'R': 16,
    'T': 17,
    'V': 18,
    'Z': 19,
    '\\': 20,
    'b': 21,
    'd': 22,
    'f': 23,
    'h': 24,
    'l': 25,
    'n': 26,
    'r': 27,
    't': 28,
    '#': 29,
    '%': 30,
    ')': 31,
    '+': 32,
    '-': 33,
    '/': 34,
    '1': 35,
    '3': 36,
    '5': 37,
    '7': 38,
    '9': 39,
    '=': 40,
    'A': 41,
    'C': 42,
    'E': 43,
    'G': 44,
    'I': 45,
    'K': 46,
    'M': 47,
    'O': 48,
    'S': 49,
    'U': 50,
    'W': 51,
    'Y': 52,
    '[': 53,
    ']': 54,
    'a': 55,
    'c': 56,
    'e': 57,
    'g': 58,
    'i': 59,
    'm': 60,
    'o': 61,
    's': 62,
    'u': 63,
    'y': 64}

def integer_label_protein(sequence, max_length=1200):
    encoding = np.zeros(max_length)
    for idx, letter in enumerate(sequence[:max_length]):
        try:
            letter = letter.upper()
            encoding[idx] = CHARPROTSET[letter]

        except KeyError:
            logging.warning(
                f"character {letter} does not exists in sequence category encoding, skip and treat as " f"padding."
            )
    return encoding
def integer_label_drug(sequence, max_length=1160):
    encoding = np.zeros(max_length)
    for idx, letter in enumerate(sequence[:max_length]):
        try:
            letter = letter.upper()
            encoding[idx] = CHARISOSMISET[letter]

        except KeyError:
            logging.warning(
                f"character {letter} does not exists in sequence category encoding, skip and treat as " f"padding."
            )
    return encoding

