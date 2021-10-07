# Embedded Training using Baum-Welch Algortihm

## Course Project
- COSE362, Machine Learning

### Input
- trainig data and initial HMM models

### Output
- Maximum likelihood HMM models for the training data


### Summary
1. Read the transcription (txt.file)
2. Construct an utterance HMM according to the transcription file
3. Run the statistics accumulation procedure of the Baum-Welch alogrithm
4. Run the model update procedure of the Baum-Welch algortihm
5. Compute the average log likelihood of the training data given the new HMMS
6. Run a speech recognizer to produce a confusion matrix
7. Increase the number of Gaussian density functions after every 5th iteration.
