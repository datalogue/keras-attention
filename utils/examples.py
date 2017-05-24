import numpy as np

EXAMPLES = ['26th January 2016', '3 April 1989', '5 Dec 09', 'Sat 8 Jun 2017']

def run_example(model, input_vocabulary, output_vocabulary, text):
    encoded = input_vocabulary.string_to_int(text)
    prediction = model.predict(np.array([encoded]))
    prediction = np.argmax(prediction[0], axis=-1)
    return output_vocabulary.int_to_string(prediction)

def run_examples(model, input_vocabulary, output_vocabulary, examples=EXAMPLES):
    predicted = []
    for example in examples:
        print('~~~~~')
        predicted.append(run_example(model, input_vocabulary, output_vocabulary, example))
        print('input:',example)
        print('output:',predicted[-1])
    return predicted