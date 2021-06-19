#  START HERE:
# 1. Set-ExecutionPolicy Unrestricted -Scope Process

# conda create -n env_pytorch python=3.6
# conda activate env_pytorch
# pip install transformers[torch]


# 2. py -m venv env
# 3. env\Scripts\activate
# 3. pip install Flask
# 4. set FLASK_APP=app.py

# 6. pip install transformers[torch] (inside venv)
# note: to execute venv just type exit or ctrl^d
from transformers import BertTokenizer
from transformers import BertForQuestionAnswering
import torch
from flask import Flask
app = Flask(__name__)
app.run(debug=True)

# Model
model = BertForQuestionAnswering.from_pretrained(
    'bert-large-uncased-whole-word-masking-finetuned-squad')

# Tokenizer
tokenizer = BertTokenizer.from_pretrained(
    'bert-large-uncased-whole-word-masking-finetuned-squad')


def answer_question(question, answer_text):
    '''
    Takes a `question` string and an `answer_text` string (which contains the
    answer), and identifies the words within the `answer_text` that are the
    answer. Prints them out.
    '''
    # ======== Tokenize ========
    # Apply the tokenizer to the input text, treating them as a text-pair.
    input_ids = tokenizer.encode(question, answer_text)

    # Report how long the input sequence is.
    print('Query has {:,} tokens.\n'.format(len(input_ids)))

    # ======== Set Segment IDs ========
    # Search the input_ids for the first instance of the `[SEP]` token.
    sep_index = input_ids.index(tokenizer.sep_token_id)

    # The number of segment A tokens includes the [SEP] token istelf.
    num_seg_a = sep_index + 1

    # The remainder are segment B.
    num_seg_b = len(input_ids) - num_seg_a

    # Construct the list of 0s and 1s.
    segment_ids = [0]*num_seg_a + [1]*num_seg_b

    # There should be a segment_id for every input token.
    assert len(segment_ids) == len(input_ids)

    # ======== Evaluate ========
    # Run our example question through the model.
    start_scores, end_scores = model(torch.tensor([input_ids]),  # The tokens representing our input text.
                                     token_type_ids=torch.tensor([segment_ids]), return_dict=False)  # The segment IDs to differentiate question from answer_text

    # ======== Reconstruct Answer ========
    # Find the tokens with the highest `start` and `end` scores.
    answer_start = torch.argmax(start_scores)
    answer_end = torch.argmax(end_scores)

    # Get the string versions of the input tokens.
    tokens = tokenizer.convert_ids_to_tokens(input_ids)

    answer = full_sentence(tokens, answer_start, answer_end)

    print('Answer: "' + answer + '"')


def start_of_sentence(tokens, answer_start, answer_end):

    # Start with the first token and/or first letter in sentence
    if tokens[answer_start-1] == '.' or tokens[answer_start] == None:
        answer = tokens[answer_start]

    # Otherwise keep searching for the beginning of the sentence
    else:
        while tokens[answer_start] != '.':
            answer_start -= 1

    # Find the last letter in the sentence
    if tokens[answer_end] != '.' or tokens[answer_end] != None:
        while tokens[answer_end] != '.' or answer_end+1 >= len(tokens):
            answer_end += 1

    return tokens[answer_start], answer_start, answer_end


def update_tokens(answer, tokens, answer_start, answer_end):

    # Select the remaining answer tokens and join them with whitespace.
    for i in range(answer_start + 1, answer_end + 1):

        # If it's a subword token, then recombine it with the previous token.
        if tokens[i][0:2] == '##':
            answer += tokens[i][2:]

        # If there is a punctuation mark, then combine without space
        elif tokens[i].isalnum() == False:
            answer += tokens[i]

        # Otherwise, add a space then the token.
        else:
            answer += ' ' + tokens[i]

    return answer


def update_punctuation(answer):

    # Set beginning of sentence to a letter
    while answer[0].isalnum() == False:
        answer = answer[1:]

    # Set end of sentence to a letter
    len_answer = len(answer)
    while answer[len_answer-1].isalnum() == False:
        answer = answer[:len_answer-1]
        len_answer -= 1

    return answer


def full_sentence(tokens, answer_start, answer_end):

    # Update indices to include full sentence
    new_res = start_of_sentence(tokens, answer_start, answer_end)
    answer = new_res[0]
    answer_start = new_res[1]
    answer_end = new_res[2]

    # Update answer to include correct spacing with punctuation and combine ## tokens
    answer = update_tokens(answer, tokens, answer_start, answer_end)

    # Remove extraneous punctuation from start and end
    answer = update_punctuation(answer)

    # Capitalize the first letter and add period punctuation to end
    return answer[0].upper()+answer[1:]+'.'


bert_abstract = "Purchased Content. When you purchase an item of content, your content will be stored in a digital locker and you may view it an unlimited number of times for during your Locker Period. The “Locker Period” will be for at least 5 years from the date of your purchase (subject to the restrictions described in the YouTube Paid Service Terms of Service). Each item of purchased content may have a different Locker Period and you agree to the Locker Period before you order it. Pausing, stopping, or rewinding purchased content will not extend the Locker Period.  As noted in the YouTube Paid Service Terms of Service, if an item of purchased content becomes unavailable during the five year period from the purchase date, you may request a refund."

question = "Can I get a refund?"

answer_question(question, bert_abstract)


print("hello world")
