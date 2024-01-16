
# Optical Character Recognition (OCR) System
Julia Donato (judonato)

## Overview
This project implements an Optical Character Recognition (OCR) system using Hidden Markov Models (HMMs). The primary objective is to recognize text from noisy images, assuming the images contain English words and sentences in a fixed-width font.

## Problem Formulation
The OCR task is formulated as a problem of probabilistic inference, where each character in an image corresponds to an observed variable, and the true character it represents is a hidden variable. The system aims to determine the most probable sequence of hidden variables (characters) given the observed data (images of characters).

### Components
1. **Image Processing**: Converts images into a format suitable for character recognition.
2. **Probability Estimation**: Calculates initial, transition, and emission probabilities.
3. **Recognition Algorithms**:
   - Simple Bayes Model: Uses emission probabilities to recognize characters.
   - HMM with Viterbi Algorithm: Employs a more sophisticated approach considering both the emission and transition probabilities.

## Implementation Details
- **Language & Libraries**: The system is implemented in Python, utilizing the PIL library for image processing.
- **Character Dimensions**: Each character is within a 14x25 pixel box.
- **Training Data**: The system learns character representations and language statistics from a provided text file.

### Key Functions
- `load_letters`: Processes image files to extract character representations.
- `estimate_initial_probabilities`: Calculates the likelihood of each character being the first in a line.
- `estimate_transition_probabilities`: Determines the probability of transitioning from one character to another.
- `estimate_emission_probabilities`: Computes the probability of each character representation given the actual character, factoring in noise.
- `recognize_simple_bayes`: Implements the simple Bayes recognition approach.
- `recognize_hmm_viterbi`: Applies the HMM with the Viterbi algorithm for character recognition.

## Challenges & Assumptions
- **Noise Handling**: A significant challenge was managing the noise in the image data. A noise level assumption was made to model the likelihood of pixel mismatches.
- **Character Width**: Initially set to 14 pixels, this parameter was critical in correctly loading and processing the image data.
- **Probabilistic Models**: Building accurate models for initial, transition, and emission probabilities was crucial and challenging.
- **Viterbi Algorithm**: Implementing and fine-tuning the Viterbi algorithm to work effectively with our probabilistic models was a complex task.

## Design Decisions & Simplifications
- **Logarithmic Probabilities**: To handle underflow issues, probabilities were managed in logarithmic form.
- **Character Set Limitation**: The system was designed to recognize a limited set of characters (uppercase and lowercase English letters, digits, and some punctuation marks), which simplified the problem but might limit its application.
- **Noise Model**: The noise model was simplified to treat each pixel independently, which may not always hold true in real-world scenarios.

## Conclusion
The OCR system demonstrates the practical application of HMMs in solving real-world problems like character recognition in noisy images. Despite some simplifications and challenges, it provides a foundational understanding of how probabilistic models can be employed in image processing and pattern recognition tasks.
