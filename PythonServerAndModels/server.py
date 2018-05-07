
from flask import Flask, request, jsonify, send_file
from model import *

app = Flask(__name__)

@app.route("/",  methods=['POST', 'GET'])
def root():
    if request.method == "POST":
        return "This is the server! (POST)"
    if request.method == "GET":
        return "This is the server! (GET)"

@app.route("/manea", methods=['POST', 'GET'])
def manea():
    print(request.method)

    dict = {}
    message = "";

    if (request.method == "GET"):
        outputLength = request.args.get('outputLength')
        pattern = request.args.get('pattern')
        author = request.args.get('author')
    elif (request.method == "POST"):
        outputLength = request.form['outputLength']
        pattern = request.form['pattern']
        author = request.form['author']

    if (outputLength == None):
        message += "Output length is None\n"
    else:
        dict['outputLength'] = outputLength

    if (pattern == None):
        message += "Pattern argument is None\n"
    else:
        dict['pattern'] = pattern

    if (author == None):
        message += "Author length is None\n"
    else:
        dict['author'] = author

    print(dict)

    manea = getManea(chars=chars,model=model, **dict)
    status = 200

    if "Couldn't" in manea:
        status=403

    return jsonify( manea=manea, message=message ), status


@app.route("/instrumental", methods=['POST', 'GET'])
def instrumental():
    prediction_output = generateNotes(model, network_input, pitchnames, notesVocab)
    path = createMidi(prediction_output)

    print("This is the path")
    print(path)

    return(send_file(filename_or_fp = path, mimetype="audio/midi", as_attachment=True))


_, text = loadManele()
text = sanitizare_manele(text)
chars = sorted(list(set(text)))

char_to_int = dict((c, i) for i, c in enumerate(chars))
int_to_char = dict((i, c) for i, c in enumerate(chars))

X, y, dataX, dataY = getManeleSequences(text,char_to_int,chars)
model = getModel(X, y)

#---------------------------------------------------------------------------> text generator

notes = getNotes()
pitchnames = sorted(set(item for item in notes))
notesVocab = len(set(notes))
network_input, normalized_input = prepareNoteSequences(notes, pitchnames, notesVocab)
model_manele = getModelInstrumental(normalized_input, notesVocab)


if __name__ == "__main__":
    dict = {}
    manea = getManea(chars=chars,model=model, **dict)
    print(manea)

    app.run(host='0.0.0.0')
