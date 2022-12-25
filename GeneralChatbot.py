import numpy as np
import tensorflow as tf
import re
import time

#******** Data Preprecessing ********#


#Importing the dataset
m_lines = open('movie_lines.txt', encoding='utf-8', errors='ignore').read().split('\n')
m_conversations = open('movie_conversations.txt', encoding='utf-8', errors='ignore').read().split('\n')

# Creating a dictionary to map each lines and  ids
id_to_line = {}
for line in m_lines:
    _line = line.split(' +++$+++ ')
    if len(_line) == 5:
        id_to_line[_line[0]] = _line[4]

#Creating a list of all of the conversations
conversationIDs = []
for conv in m_conversations[:-1]:
    _conv = conv.split(' +++$+++ ')[-1][1:-1].replace("'", "").replace(" ", "")
    conversationIDs.append(_conv.split(','))

#Separating the questions and the answers
Qs = []
ANSs = []
for conv in conversationIDs:
    for i in range(len(conv) - 1):
        Qs.append(id_to_line[conv[i]])
        ANSs.append(id_to_line[conv[i + 1]])


#first step of text cleaning
def textCleaning(txt):
    txt = txt.lower()
    txt = re.sub(r"she's", "she is", txt)
    txt = re.sub(r"he's", "he is", txt)
    txt = re.sub(r"i'm", "i am", txt)
    txt = re.sub(r"that's", "that is", txt)
    txt = re.sub(r"where's", "where is", txt)
    txt = re.sub(r"what's", "what is", txt)
    txt = re.sub(r"how's", "how is", txt)
    txt = re.sub(r"\'ve", " have", txt)
    txt = re.sub(r"\'ll", " will", txt)
    txt = re.sub(r"\'re", " are", txt)
    txt = re.sub(r"n't", " not", txt)
    txt = re.sub(r"\'d", " would", txt)
    txt = re.sub(r"can't", "cannot", txt)
    txt = re.sub(r"won't", "will not", txt)
    txt = re.sub(r"[-()\"#/@;:<>{}`+=~|.!?,]", "", txt)
    return txt


#Cleaning the questions and answers
cleaned_Qs = []
for q in Qs:
    cleaned_Qs.append(textCleaning(q))


cleaned_ANSs = []
for answr in ANSs:
    cleaned_ANSs.append(textCleaning(answr))

#Removing the too short or too long questions and their answers
short_Qs = []
short_ANSs = []
j = 0
for q in cleaned_Qs:
    if 2 <= len(q.split()) <= 25:
        short_Qs.append(q)
        short_ANSs.append(cleaned_ANSs[j])
    j += 1
cleaned_Qs = []
cleaned_ANSs = []
j = 0
for answr in short_ANSs:
    if 2 <= len(answr.split()) <= 25:
        cleaned_ANSs.append(answr)
        cleaned_Qs.append(short_Qs[j])
    j += 1

#Creating a dictionary to map each word to its frequency
wrd2cnt = {}
for q in cleaned_Qs:
    for wrd in q.split():
        if wrd not in wrd2cnt:
            wrd2cnt[wrd] = 1
        else:
            wrd2cnt[wrd] += 1
for answr in cleaned_ANSs:
    for wrd in answr.split():
        if wrd not in wrd2cnt:
            wrd2cnt[wrd] = 1
        else:
            wrd2cnt[wrd] += 1

#Creating two dictionaries to assign a unique integer to the questions/answers words
thrshld_Qs = 15
Qs_wrds2int = {}
wrd_num = 0
for wrd, cnt in wrd2cnt.items():
    if cnt >= thrshld_Qs:
        Qs_wrds2int[wrd] = wrd_num
        wrd_num += 1
thrshld_ANSs = 15
ANSs_wrds2int = {}
wrd_num = 0
for wrd, cnt in wrd2cnt.items():
    if cnt >= thrshld_ANSs:
        ANSs_wrds2int[wrd] = wrd_num
        wrd_num += 1

#Adding four tokens to the two created dictionaries
tokens = ['<PAD>', '<EOS>', '<OUT>', '<SOS>']
for token in tokens:
    Qs_wrds2int[token] = len(Qs_wrds2int) + 1
for token in tokens:
    ANSs_wrds2int[token] = len(ANSs_wrds2int) + 1

#Creating the inverse dictionary of the answerswords2int dictionary
ANSs_ints2wrd = {w_i: w for w, w_i in ANSs_wrds2int.items()}

#Adding the End Of String token to the end of every answer
for i in range(len(cleaned_ANSs)):
    cleaned_ANSs[i] += ' <EOS>'

#Translating all the questions and the answers into integers and Replacing all the words that were filtered out by <OUT>
Qs_into_int = []
for q in cleaned_Qs:
    ints = []
    for wrd in q.split():
        if wrd not in Qs_wrds2int:
            ints.append(Qs_wrds2int['<OUT>'])
        else:
            ints.append(Qs_wrds2int[wrd])
    Qs_into_int.append(ints)
ANSs_into_int = []
for answr in cleaned_ANSs:
    ints = []
    for wrd in answr.split():
        if wrd not in ANSs_wrds2int:
            ints.append(ANSs_wrds2int['<OUT>'])
        else:
            ints.append(ANSs_wrds2int[wrd])
    ANSs_into_int.append(ints)

#Sorting questions and answers by the length of questions and excluding the long ones (more than 25 words)
# (to speed up the training & help to reduce the loss)
sorted_cleaned_Qs = []
sorted_cleaned_ANSs = []
for length in range(1, 25 + 1):
    for i in enumerate(Qs_into_int):
        if len(i[1]) == length:
            sorted_cleaned_Qs.append(Qs_into_int[i[0]])
            sorted_cleaned_ANSs.append(ANSs_into_int[i[0]])


#******** SEQ2SEQ Model Building *********#


#Creating placeholders for the inputs and the targets
def model_inputs():
    inputs = tf.placeholder(tf.int32, [None, None], name='input')
    targets = tf.placeholder(tf.int32, [None, None], name='target')
    lr = tf.placeholder(tf.float32, name='learning_rate')
    keep_prob = tf.placeholder(tf.float32, name='keep_prob')
    return inputs, targets, lr, keep_prob


#Preprocessing the targets
def preprocess_targets(targets, wrd2int, batch_size):
    left_side = tf.fill([batch_size, 1], wrd2int['<SOS>'])
    right_side = tf.strided_slice(targets, [0, 0], [batch_size, -1], [1, 1])
    preprocessed_targets = tf.concat([left_side, right_side], 1)
    return preprocessed_targets


#Creating the Encoder RNN
def encoder_rnn(rnn_inputs, rnn_size, num_layers, keep_prob, sequence_length):
    lstm = tf.contrib.rnn.BasicLSTMCell(rnn_size)
    lstm_dropout = tf.contrib.rnn.DropoutWrapper(lstm, input_keep_prob=keep_prob)
    encoder_cell = tf.contrib.rnn.MultiRNNCell([lstm_dropout] * num_layers)
    encoder_output, encoder_state = tf.nn.bidirectional_dynamic_rnn(cell_fw=encoder_cell,
                                                                    cell_bw=encoder_cell,
                                                                    sequence_length=sequence_length,
                                                                    inputs=rnn_inputs,
                                                                    dtype=tf.float32)
    return encoder_state


#Decoding the training set
def decode_training_set(encoder_state, decoder_cell, decoder_embedded_input, sequence_length, decoding_scope, output_function, keep_prob, batch_size):
    attention_states = tf.zeros([batch_size, 1, decoder_cell.output_size])
    attention_keys, attention_values, attention_score_function, attention_construct_function = tf.contrib.seq2seq.prepare_attention(
        attention_states, attention_option="bahdanau", num_units=decoder_cell.output_size)
    training_decoder_function = tf.contrib.seq2seq.attention_decoder_fn_train(encoder_state[0],
                                                                              attention_keys,
                                                                              attention_values,
                                                                              attention_score_function,
                                                                              attention_construct_function,
                                                                              name="attn_dec_train")
    decoder_output, decoder_final_state, decoder_final_context_state = tf.contrib.seq2seq.dynamic_rnn_decoder(
        decoder_cell,
        training_decoder_function,
        decoder_embedded_input,
        sequence_length,
        scope=decoding_scope)
    decoder_output_dropout = tf.nn.dropout(decoder_output, keep_prob)
    return output_function(decoder_output_dropout)


#Decoding the test/validation set
def decode_test_set(encoder_state, decoder_cell, decoder_embeddings_matrix, sos_id, eos_id, maximum_length, num_wrds,
                    decoding_scope, output_function, keep_prob, batch_size):
    attention_states = tf.zeros([batch_size, 1, decoder_cell.output_size])
    attention_keys, attention_values, attention_score_function, attention_construct_function = tf.contrib.seq2seq.prepare_attention(
        attention_states, attention_option="bahdanau", num_units=decoder_cell.output_size)
    test_decoder_function = tf.contrib.seq2seq.attention_decoder_fn_inference(output_function,
                                                                              encoder_state[0],
                                                                              attention_keys,
                                                                              attention_values,
                                                                              attention_score_function,
                                                                              attention_construct_function,
                                                                              decoder_embeddings_matrix,
                                                                              sos_id,
                                                                              eos_id,
                                                                              maximum_length,
                                                                              num_wrds,
                                                                              name="attn_dec_inf")
    test_predictions, decoder_final_state, decoder_final_context_state = tf.contrib.seq2seq.dynamic_rnn_decoder(
        decoder_cell,
        test_decoder_function,
        scope=decoding_scope)
    return test_predictions


#Creating the Decoder RNN
def decoder_rnn(decoder_embedded_input, decoder_embeddings_matrix, encoder_state, num_wrds, sequence_length, rnn_size,
                num_layers, wrd2int, keep_prob, batch_size):
    with tf.variable_scope("decoding") as decoding_scope:
        lstm = tf.contrib.rnn.BasicLSTMCell(rnn_size)
        lstm_dropout = tf.contrib.rnn.DropoutWrapper(lstm, input_keep_prob=keep_prob)
        decoder_cell = tf.contrib.rnn.MultiRNNCell([lstm_dropout] * num_layers)
        weights = tf.truncated_normal_initializer(stddev=0.1)
        biases = tf.zeros_initializer()
        output_function = lambda x: tf.contrib.layers.fully_connected(x,
                                                                      num_wrds,
                                                                      None,
                                                                      scope=decoding_scope,
                                                                      weights_initializer=weights,
                                                                      biases_initializer=biases)
        training_predictions = decode_training_set(encoder_state,
                                                   decoder_cell,
                                                   decoder_embedded_input,
                                                   sequence_length,
                                                   decoding_scope,
                                                   output_function,
                                                   keep_prob,
                                                   batch_size)
        decoding_scope.reuse_variables()
        test_predictions = decode_test_set(encoder_state,
                                           decoder_cell,
                                           decoder_embeddings_matrix,
                                           wrd2int['<SOS>'],
                                           wrd2int['<EOS>'],
                                           sequence_length - 1,
                                           num_wrds,
                                           decoding_scope,
                                           output_function,
                                           keep_prob,
                                           batch_size)
    return training_predictions, test_predictions


#Building the seq2seq model
def seq2seq_model(inputs, targets, keep_prob, batch_size, sequence_length, ANSs_num_wrds, Qs_num_wrds,
                  encoder_embedding_size, decoder_embedding_size, rnn_size, num_layers, Qs_wrds2int):
    encoder_embedded_input = tf.contrib.layers.embed_sequence(inputs,
                                                              ANSs_num_wrds + 1,
                                                              encoder_embedding_size,
                                                              initializer=tf.random_uniform_initializer(0, 1))
    encoder_state = encoder_rnn(encoder_embedded_input, rnn_size, num_layers, keep_prob, sequence_length)
    preprocessed_targets = preprocess_targets(targets, Qs_wrds2int, batch_size)
    decoder_embeddings_matrix = tf.Variable(tf.random_uniform([Qs_num_wrds + 1, decoder_embedding_size], 0, 1))
    decoder_embedded_input = tf.nn.embedding_lookup(decoder_embeddings_matrix, preprocessed_targets)
    training_predictions, test_predictions = decoder_rnn(decoder_embedded_input,
                                                         decoder_embeddings_matrix,
                                                         encoder_state,
                                                         Qs_num_wrds,
                                                         sequence_length,
                                                         rnn_size,
                                                         num_layers,
                                                         Qs_wrds2int,
                                                         keep_prob,
                                                         batch_size)
    return training_predictions, test_predictions


#******** Model Training *********#


epochs = 100
batch_size = 32
rnn_size = 1024
num_layers = 3
encoding_embedding_size = 1024
decoding_embedding_size = 1024
learning_rate = 0.001
learning_rate_decay = 0.9
min_learning_rate = 0.0001
keep_probability = 0.5

tf.reset_default_graph()
session = tf.InteractiveSession()

inputs, targets, lr, keep_prob = model_inputs()

sequence_length = tf.placeholder_with_default(25, None, name='sequence_length')

input_shape = tf.shape(inputs)

# training and test predictions
training_predictions, test_predictions = seq2seq_model(tf.reverse(inputs, [-1]),
                                                       targets,
                                                       keep_prob,
                                                       batch_size,
                                                       sequence_length,
                                                       len(ANSs_wrds2int),
                                                       len(Qs_wrds2int),
                                                       encoding_embedding_size,
                                                       decoding_embedding_size,
                                                       rnn_size,
                                                       num_layers,
                                                       Qs_wrds2int)

# Setting up the Loss Error, the Optimizer and Gradient Clipping
with tf.name_scope("optimization"):
    loss_error = tf.contrib.seq2seq.sequence_loss(training_predictions,
                                                  targets,
                                                  tf.ones([input_shape[0], sequence_length]))
    optimizer = tf.train.AdamOptimizer(learning_rate)
    gradients = optimizer.compute_gradients(loss_error)
    clipped_gradients = [(tf.clip_by_value(grad_tensor, -5., 5.), grad_variable) for grad_tensor, grad_variable in
                         gradients if grad_tensor is not None]
    optimizer_gradient_clipping = optimizer.apply_gradients(clipped_gradients)


# Padding the sequences with the <PAD> token
def apply_padding(batch_of_sequences, wrd2int):
    max_sequence_length = max([len(sequence) for sequence in batch_of_sequences])
    return [sequence + [wrd2int['<PAD>']] * (max_sequence_length - len(sequence)) for sequence in batch_of_sequences]


#Splitting the data into batches of questions and answers
def split_into_batches(Qs, ANSs, batch_size):
    for batch_index in range(0, len(Qs) // batch_size):
        start_index = batch_index * batch_size
        Qs_in_batch = Qs[start_index: start_index + batch_size]
        ANSs_in_batch = ANSs[start_index: start_index + batch_size]
        padded_Qs_in_batch = np.array(apply_padding(Qs_in_batch, Qs_wrds2int))
        padded_ANSs_in_batch = np.array(apply_padding(ANSs_in_batch, ANSs_wrds2int))
        yield padded_Qs_in_batch, padded_ANSs_in_batch


#Splitting the questions and answers into training and validation sets
training_validation_split = int(len(sorted_cleaned_Qs) * 0.15)
training_Qs = sorted_cleaned_Qs[training_validation_split:]
training_ANSs = sorted_cleaned_ANSs[training_validation_split:]
validation_Qs = sorted_cleaned_Qs[:training_validation_split]
validation_ANSs = sorted_cleaned_ANSs[:training_validation_split]

#Training
batch_index_check_training_loss = 100
batch_index_check_validation_loss = ((len(training_Qs)) // batch_size // 2) - 1
total_training_loss_error = 0
list_validation_loss_error = []
early_stopping_check = 0
early_stopping_stop = 100
checkpoint = "chatbot_weights.ckpt"
session.run(tf.global_variables_initializer())
for epoch in range(1, epochs + 1):
    for batch_index, (padded_Qs_in_batch, padded_ANSs_in_batch) in enumerate(
            split_into_batches(training_Qs, training_ANSs, batch_size)):
        starting_time = time.time()
        _, batch_training_loss_error = session.run([optimizer_gradient_clipping, loss_error],
                                                   {inputs: padded_Qs_in_batch,
                                                    targets: padded_ANSs_in_batch,
                                                    lr: learning_rate,
                                                    sequence_length: padded_ANSs_in_batch.shape[1],
                                                    keep_prob: keep_probability})
        total_training_loss_error += batch_training_loss_error
        ending_time = time.time()
        batch_time = ending_time - starting_time
        if batch_index % batch_index_check_training_loss == 0:
            print(
                'Epoch: {:>3}/{}, Batch: {:>4}/{}, Training Loss Error: {:>6.3f}, Training Time on 100 Batches: {:d} seconds'.format(
                    epoch,
                    epochs,
                    batch_index,
                    len(training_Qs) // batch_size,
                    total_training_loss_error / batch_index_check_training_loss,
                    int(batch_time * batch_index_check_training_loss)))
            total_training_loss_error = 0
        if batch_index % batch_index_check_validation_loss == 0 and batch_index > 0:
            total_validation_loss_error = 0
            starting_time = time.time()
            for batch_index_validation, (padded_Qs_in_batch, padded_ANSs_in_batch) in enumerate(
                    split_into_batches(validation_Qs, validation_ANSs, batch_size)):
                batch_validation_loss_error = session.run(loss_error, {inputs: padded_Qs_in_batch,
                                                                       targets: padded_ANSs_in_batch,
                                                                       lr: learning_rate,
                                                                       sequence_length: padded_ANSs_in_batch.shape[
                                                                           1],
                                                                       keep_prob: 1})
                total_validation_loss_error += batch_validation_loss_error
            ending_time = time.time()
            batch_time = ending_time - starting_time
            average_validation_loss_error = total_validation_loss_error / (len(validation_Qs) / batch_size)
            print('Validation Loss Error: {:>6.3f}, Batch Validation Time: {:d} seconds'.format(
                average_validation_loss_error, int(batch_time)))
            learning_rate *= learning_rate_decay
            if learning_rate < min_learning_rate:
                learning_rate = min_learning_rate
            list_validation_loss_error.append(average_validation_loss_error)
            if average_validation_loss_error <= min(list_validation_loss_error):
                print('I speak better now!!')
                early_stopping_check = 0
                saver = tf.train.Saver()
                saver.save(session, checkpoint)
            else:
                print("I apologize for my poor speech; I need to work on it more.")
                early_stopping_check += 1
                if early_stopping_check == early_stopping_stop:
                    break
    if early_stopping_check == early_stopping_stop:
        print("Sorry, I can't speak much better now. The best I can speak is this..")
        break
print("Game Over")

#******** Model Testing *********#


# Loading the weights and Running the session
checkpoint = "./chatbot_weights.ckpt"
session = tf.InteractiveSession()
session.run(tf.global_variables_initializer())
saver = tf.train.Saver()
saver.restore(session, checkpoint)


# Converting the questions from strings to lists of encoding integers
def convrt_str2int(qstn, wrd2int):
    qstn = textCleaning(qstn)
    return [wrd2int.get(wrd, wrd2int['<OUT>']) for wrd in qstn.split()]


# Setting up the chat
while (True):
    qstn = input("You: ")
    if qstn == 'Goodbye':
        break
    qstn = convrt_str2int(qstn, Qs_wrds2int)
    qstn = qstn + [Qs_wrds2int['<PAD>']] * (25 - len(qstn))
    fake_batch = np.zeros((batch_size, 25))
    fake_batch[0] = qstn
    predicted_ans = session.run(test_predictions, {inputs: fake_batch, keep_prob: 0.5})[0]
    answr = ''
    for i in np.argmax(predicted_ans, 1):
        if ANSs_ints2wrd[i] == 'i':
            token = ' I'
        elif ANSs_ints2wrd[i] == '<EOS>':
            token = '.'
        elif ANSs_ints2wrd[i] == '<OUT>':
            token = 'out'
        else:
            token = ' ' + ANSs_ints2wrd[i]
        answr += token
        if token == '.':
            break
    print('ChatBot: ' + answr)
