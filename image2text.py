#!/usr/bin/python
#
# Perform optical character recognition, usage:
#     python3 ./image2text.py train-image-file.png train-text.txt test-image-file.png
#
# Authors: Julia Donato judonato
# (based on skeleton code by D. Crandall, Nov 2023)
#

from PIL import Image
import sys
import math
from collections import defaultdict

CHARACTER_WIDTH=14
CHARACTER_HEIGHT=25
TRAIN_LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789(),.-!?\"' "


def load_letters(fname):
    im = Image.open(fname)
    px = im.load()
    (x_size, y_size) = im.size
    print(im.size)
    print(int(x_size / CHARACTER_WIDTH) * CHARACTER_WIDTH)
    result = []
    for x_beg in range(0, int(x_size / CHARACTER_WIDTH) * CHARACTER_WIDTH, CHARACTER_WIDTH):
        result += [ [ "".join([ '*' if px[x, y] < 1 else ' ' for x in range(x_beg, x_beg+CHARACTER_WIDTH) ]) for y in range(0, CHARACTER_HEIGHT) ], ]
    return result

def load_training_letters(fname):
    TRAIN_LETTERS="ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789(),.-!?\"' "
    letter_images = load_letters(fname)
    return { TRAIN_LETTERS[i]: letter_images[i] for i in range(0, len(TRAIN_LETTERS) ) }


def estimate_initial_probabilities(train_txt_fname):
    initial_prob = defaultdict(int)
    with open(train_txt_fname, 'r') as file:
        for line in file:
            initial_prob[line[0]] += 1
    total = sum(initial_prob.values())
    for key in initial_prob:
        initial_prob[key] = math.log(initial_prob[key] / total)
    return initial_prob

def estimate_transition_probabilities(train_txt_fname):
    transition_prob = defaultdict(lambda: defaultdict(int))
    with open(train_txt_fname, 'r') as file:
        for line in file:
            for i in range(len(line) - 1):
                transition_prob[line[i]][line[i+1]] += 1
    for prev_char in transition_prob:
        total = sum(transition_prob[prev_char].values())
        for char in transition_prob[prev_char]:
            transition_prob[prev_char][char] = math.log(transition_prob[prev_char][char] / total)
    return transition_prob

def estimate_emission_probabilities(train_letters, test_letters, noise_level=0.5):
    emission_prob = defaultdict(lambda: defaultdict(float))
    for char in TRAIN_LETTERS:
        train_char_image = train_letters[char]
        for i, test_char_image in enumerate(test_letters):
            match_count = 0
            total_pixels = CHARACTER_WIDTH * CHARACTER_HEIGHT
            for x in range(CHARACTER_WIDTH):
                for y in range(CHARACTER_HEIGHT):
                    if test_char_image[y][x] == train_char_image[y][x]:
                        match_count += 1
            
            # Calculate the probability of observing the test character given the train character
            match_prob = (100 - noise_level) / 100
            no_match_prob = noise_level / 100
            matched_pixels_prob = match_prob ** match_count
            unmatched_pixels_prob = no_match_prob ** (total_pixels - match_count)
            # Use logarithms to avoid underflow in multiplication of probabilities
            emission_prob[i][char] = math.log(matched_pixels_prob * unmatched_pixels_prob)
    return emission_prob

def recognize_simple_bayes(test_letters, initial_prob, emission_prob):
    recognized_text = ""
    for i, test_letter in enumerate(test_letters):
        max_prob = float('-inf')
        best_char = ''
        for char in TRAIN_LETTERS:
            # Simple model considers only the emission probability
            char_prob = emission_prob[i][char]
            if char_prob > max_prob:
                max_prob = char_prob
                best_char = char
        recognized_text += best_char
    return recognized_text

def recognize_hmm_viterbi(test_letters, initial_prob, transition_prob, emission_prob):
    states = TRAIN_LETTERS
    V = [{}]
    path = {}

    # Initialize base cases (t == 0)
    for state in states:
        V[0][state] = initial_prob.get(state, float('-inf')) + emission_prob[0].get(state, float('-inf'))
        path[state] = [state]

    # Run Viterbi for t > 0
    for t in range(1, len(test_letters)):
        V.append({})
        new_path = {}

        for curr_state in states:
            max_prob, max_state = max(
                (V[t-1][prev_state] + transition_prob[prev_state].get(curr_state, float('-inf')), prev_state)
                for prev_state in states
            )
            V[t][curr_state] = max_prob + emission_prob[t].get(curr_state, float('-inf'))
            new_path[curr_state] = path[max_state] + [curr_state]

        path = new_path

    # Backtrack to find the most probable path
    n = len(test_letters)
    last_state = max(V[n-1], key=V[n-1].get)
    return ''.join(path[last_state])


# Main program
if len(sys.argv) != 4:
    raise Exception("Usage: python3 ./image2text.py train-image-file.png train-text.txt test-image-file.png")

(train_img_fname, train_txt_fname, test_img_fname) = sys.argv[1:]
train_letters = load_training_letters(train_img_fname)
test_letters = load_letters(test_img_fname)

# Estimate the probabilities required for the models
initial_prob = estimate_initial_probabilities(train_txt_fname)
transition_prob = estimate_transition_probabilities(train_txt_fname)
emission_prob = estimate_emission_probabilities(train_letters, test_letters)

# Recognize the text using the simple Bayes net
simple_bayes_result = recognize_simple_bayes(test_letters, initial_prob, emission_prob)

# Recognize the text using the HMM with Viterbi decoding
hmm_viterbi_result = recognize_hmm_viterbi(test_letters, initial_prob, transition_prob, emission_prob)

# Print the final results
print("Simple: " + simple_bayes_result)
print("   HMM: " + hmm_viterbi_result)