from __future__ import print_function
import tensorflow as tf
from keras import backend as K
from keras.callbacks import LambdaCallback
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.callbacks import ModelCheckpoint
from keras.layers import LSTM
from keras.optimizers import RMSprop
from keras.utils import np_utils
from keras.utils.data_utils import get_file
from music21 import instrument, note, stream, chord

import numpy as np
import random
import sys
import io
import os
import re
import pickle
import time

from os import listdir

seq_length = 100

with tf.Graph().as_default():
    K.set_learning_phase(1)

def loadManele(path='./resources/manele.txt'):
    with io.open(path, encoding='utf-8') as file:text = file.read().lower() #read file

    unaltered_text = text
    dict_manele = {}
    text=text.split('\n')

    index=0
    while index < len(text)-1:
        chars = sorted(list(set(text[index])))
        if len(text[index]) == 0:
            titlu = text[index+1]
            index +=3
            manea = ""
            while index < len(text) and len(text[index]) > 0:
                if (index < len(text)):
                    manea += text[index]
                index+=1
            dict_manele[titlu] = manea
        else:
            index +=1

    return dict_manele, unaltered_text

def sanitizare_manele(text):
    regex_jmek1 = r"([\[]|[\(]|[\{][ ])+[ \n\r:\t!',-.:?\’“”…%\£$@–*/&><`–0-9A-Za-z]+[\)\]\} ]*"
    regex_jmek2 = r"[%*/&><`{}£–\n\r:\t!',-.:?’“”…\[\]\(\)\"тἰ0-9а‘]+"
    regex_jmek3 = r"[àâäă]+"
    regex_jmek4 = r"[tțţ]+"
    regex_jmek5 = r"[î]+"
    regex_jmek6 = r"[ü]+"
    regex_jmek7 = r"[ö]+"
    regex_jmek8 = r"[șş]+"
    regex_jmek9 = r"[ ]+"


    text = re.sub(regex_jmek1, " ", text)
    text = re.sub(regex_jmek2, " ", text)
    text = re.sub(regex_jmek3, "a", text)
    text = re.sub(regex_jmek4, "t", text)
    text = re.sub(regex_jmek5, "i", text)
    text = re.sub(regex_jmek6, "u", text)
    text = re.sub(regex_jmek7, "o", text)
    text = re.sub(regex_jmek8, "s", text)
    text = re.sub(regex_jmek9, " ", text)

    return text

def getManeleSequences(text='', char_to_int = [], chars =[]):
    dataX = []
    dataY = []

    for i in range(0, len(text) - seq_length, 1):
    	seq_in = text[i:i + seq_length]
    	seq_out = text[i + seq_length]
    	dataX.append([char_to_int[char] for char in seq_in])
    	dataY.append(char_to_int[seq_out])

    n_patterns = len(dataX)

    X = np.reshape(dataX, (n_patterns, seq_length, 1))
    X = X / float(len(chars))
    y = np_utils.to_categorical(dataY)

    return X,y, dataX, dataY

def getModel(X, y):
    model = Sequential()
    model.add(LSTM(256, input_shape=(X.shape[1], X.shape[2]), return_sequences=True))
    model.add(Dropout(0.3))
    model.add(LSTM(512, return_sequences=True))
    model.add(Dropout(0.4))
    model.add(LSTM(256))
    model.add(Dropout(0.2))
    model.add(Dense(y.shape[1], activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam')

    return model


def getManea(chars, model, outputLength=1000, pattern="ban pe ban pe ban pe ban in stil american", author="salam" ):
    n_vocab = len(chars)

    try:
        print("weights"+"/"+author+"/" + listdir("weights"+"/"+author+"/")[0])
        x = "weights"+"/"+author+"/" + listdir("weights"+"/"+author+"/")[0]
        print("This is manelelelelelelele")
        print(x)
        model.load_weights("weights"+"/"+author+"/" + listdir("weights"+"/"+author+"/")[0])
    except Exception as e:
        print("Couldn't load weights")
        print(e)
        return "Couldn't find load for author=" + author

    dataX = []

    if (len(pattern) < seq_length):
        print("smaller sequence")
        padding = " " * (seq_length - len(pattern))
        pattern += padding

    if (len(pattern) > seq_length):
        print("bigger sequence")
        pattern = pattern[0:seq_length]

    char_to_int = dict((c, i) for i, c in enumerate(chars))
    int_to_char = dict((i, c) for i, c in enumerate(chars))
    dataX.append([char_to_int[char] for char in pattern])

    start = np.random.randint(0, len(dataX))
    pattern = dataX[start]

    result_final = ''

    if (type(outputLength) is str):
        outputLength = int(outputLength)

    for i in range(outputLength):
        x = np.reshape(pattern, (1, len(pattern), 1))
        x = x / float(n_vocab)
        prediction = model.predict(x, verbose=0)
        index = np.argmax(prediction)
        result = int_to_char[index]
        result_final += result
        sys.stdout.write(result)
        pattern.append(index)
        pattern = pattern[1:len(pattern)]

    return result_final


def prepareNoteSequences(notes, pitchnames, n_vocab):
    """ Prepare the sequences used by the Neural Network """
    note_to_int = dict((note, number) for number, note in enumerate(pitchnames))

    sequence_length = 100
    network_input = []
    output = []
    for i in range(0, len(notes) - sequence_length, 1):
        sequence_in = notes[i:i + sequence_length]
        sequence_out = notes[i + sequence_length]
        network_input.append([note_to_int[char] for char in sequence_in])
        output.append(note_to_int[sequence_out])

    n_patterns = len(network_input)

    # reshape the input into a format compatible with LSTM layers
    normalized_input = np.reshape(network_input, (n_patterns, sequence_length, 1))
    # normalize input
    normalized_input = normalized_input / float(n_vocab)

    return (network_input, normalized_input)\

def getModelInstrumental(network_input, n_vocab):

    """ create the structure of the neural network """
    model = Sequential()
    model.add(LSTM(
        512,
        input_shape=(network_input.shape[1], network_input.shape[2]),
        return_sequences=True
    ))
    model.add(Dropout(0.3))
    model.add(LSTM(512, return_sequences=True))
    model.add(Dropout(0.3))
    model.add(LSTM(512))
    model.add(Dense(256))
    model.add(Dropout(0.3))
    model.add(Dense(n_vocab))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

    return model

def generateNotes(model, network_input, pitchnames, n_vocab):
    """ Generate notes from the neural network based on a sequence of notes """
    # pick a random sequence from the input as a starting point for the prediction
    start = np.random.randint(0, len(network_input)-1)

    int_to_note = dict((number, note) for number, note in enumerate(pitchnames))

    pattern = network_input[start]
    prediction_output = []

    # generate 500 notes
    for note_index in range(500):
        prediction_input = np.reshape(pattern, (1, len(pattern), 1))
        prediction_input = prediction_input / float(n_vocab)

        prediction = model.predict(prediction_input, verbose=0)

        index = np.argmax(prediction)
        result = int_to_note[index]
        prediction_output.append(result)

        pattern.append(index)
        pattern = pattern[1:len(pattern)]

    return prediction_output

def createMidi(prediction_output):
    """ convert the output from the prediction to notes and create a midi file
        from the notes """
    offset = 0
    output_notes = []

    # create note and chord objects based on the values generated by the model
    for pattern in prediction_output:

        # pattern is a chord
        if ('.' in pattern) or pattern.isdigit():
            notes_in_chord = pattern.split('.')
            notes = []
            for current_note in notes_in_chord:
                new_note = note.Note(int(current_note))
                new_note.storedInstrument = instrument.Piano()
                notes.append(new_note)
            new_chord = chord.Chord(notes)
            new_chord.offset = offset
            output_notes.append(new_chord)

        # pattern is a note
        else:
            new_note = note.Note(pattern)
            new_note.offset = offset
            new_note.storedInstrument = instrument.Piano()
            output_notes.append(new_note)

        # increase offset each iteration so that notes do not stack
        offset += 0.5

    midi_stream = stream.Stream(output_notes)
    song_name = str(time.time()) + '.mid'
    midi_stream.write('midi', fp='currentSongs/'+song_name)

    return  'currentSongs/' + song_name

def getNotes():
    with open('resources/notes', 'rb') as filepath: notes = pickle.load(filepath)
    return notes
