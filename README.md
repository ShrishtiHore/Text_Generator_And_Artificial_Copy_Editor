# AI based Text Generator And Copy Editor

### Code and Resources Used

**Language:** Python 3.8

**Modules and Libraries:** numpy, pandas, matplotlib, os, Tensorflow, time, date

**Dataset:** Google

**Keywords:** Text Correction, NLP, Deep Learning, Text Processing

- To create a dataset for Deep Text Corrector models, we start with a large collection of mostly grammatically correct samples of conversational written English. The primary dataset considered in this project is the Cornell Movie-Dialogs Corpus, which contains over 300k lines from movie scripts. This was the largest collection of conversational written English I could find that was mostly grammatically correct.
- Given a sample of text like this, the next step is to generate input-output pairs to be used during training. This is done by:
- Drawing a sample sentence from the dataset.
- Setting the input sequence to this sentence after randomly applying certain perturbations.-
- Setting the output sequence to the unperturbed sentence.
where the perturbations applied in step (2) are intended to introduce small grammatical errors which we would like the model to learn to correct. Thus far, these perturbations are limited to the:
1. subtraction of articles (a, an, the)
2. subtraction of the second part of a verb contraction (e.g. "'ve", "'ll", "'s", "'m")
replacement of a few common homophones with one of their counterparts (e.g. replacing "their" with "there", "then" with "than")
- The rates with which these perturbations are introduced are loosely based on figures taken from the CoNLL 2014 Shared Task on Grammatical Error Correction. In this project, each perturbation is applied in 25% of cases where it could potentially be applied.
- To artificially increase the dataset when training a sequence model, we perform the sampling strategy described above multiple times to arrive at 2-3x the number of input-output pairs. Given this augmented dataset, training proceeds in a very similar manner to TensorFlow's sequence-to-sequence tutorial. That is, we train a sequence-to-sequence model using - LSTM encoders and decoders with an attention mechanism as described in Bahdanau et al., 2014 using stochastic gradient descent.
- Character-level RNN (Recurrent Neural Net) LSTM (Long Short-Term Memory) implemented in Python 2.7/TensorFlow in order to predict a text based on a given dataset
- While the model is training it will periodically write checkpoint files to the cv folder. 
- The frequency with which these checkpoints are written is controlled with number of iterations, as specified with the eval_val_every option (e.g. if this is 1 then a checkpoint is written every iteration). 
- The filename of these checkpoints contains a very important number: the loss. 
- For example, a checkpoint with filename lm_lstm_epoch0.95_2.0681.t7 indicates that at this point the model was on epoch 0.95 (i.e. it has almost done one full pass over the training data), and the loss on validation data was 2.0681.
- This number is very important because the lower it is, the better the checkpoint works. Once you start to generate data (discussed below), you will want to use the model checkpoint that reports the lowest validation loss. 
- Notice that this might not necessarily be the last checkpoint at the end of training (due to possible overfitting).
- Another important quantities to be aware of are batch_size (call it B), seq_length (call it S), and the train_frac and val_frac settings. The batch size specifies how many streams of data are processed in parallel at one time. 
- The sequence length specifies the length of each stream, which is also the limit at which the gradients can propagate backwards in time. For example, if seq_length is 20, then the gradient signal will never backpropagate more than 20 time steps, and the model might not find dependencies longer than this length in number of characters. Thus, if you have a very difficult dataset where there are a lot of long-term dependencies you will want to increase this setting. 
- Now, if at runtime your input text file has N characters, these first all get split into chunks of size BxS. These chunks then get allocated across three splits: train/val/test according to the frac settings. 
- By default train_frac is 0.95 and val_frac is 0.05, which means that 95% of our data chunks will be trained on and 5% of the chunks will be used to estimate the validation loss (and hence the generalization). If your data is small, it's possible that with the default settings you'll only have very few chunks in total (for example 100). This is bad: 
- In these cases you may want to decrease batch size or sequence length.
