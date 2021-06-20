#  START HERE:
# 1. Set-ExecutionPolicy Unrestricted -Scope Process

# conda create -n env_pytorch python=3.6
# conda activate env_pytorch
# pip install transformers[torch]


# 2. py -m venv env
# 3. env\Scripts\activate
# 3. pip install Flask
# 4. pip install BeautifulSoup4
# 5. set FLASK_APP=app.py

# 6. pip install transformers[torch] (inside venv)
# note: to execute venv just type exit or ctrl^d
import urllib.request
from bs4.element import Comment
from bs4 import BeautifulSoup
from transformers import BertTokenizer
from transformers import BertForQuestionAnswering
import torch
from flask import Flask, jsonify, request, render_template
app = Flask(__name__)
app.run(debug=True)

# Load the home page


@app.route('/')
def home_page():
    return render_template('search.html')

# Load the home page again


# @app.route('/home_2')
# def home_page2():
#     alert("inside home2")
#     return render_template('search.html')


# When search clicked, load quote.html


@app.route('/select')
def selector():
    return render_template('quote.html')


# Model
model = BertForQuestionAnswering.from_pretrained(
    'bert-large-uncased-whole-word-masking-finetuned-squad')

# Tokenizer
tokenizer = BertTokenizer.from_pretrained(
    'bert-large-uncased-whole-word-masking-finetuned-squad')

###########################  QA FUNCTIONS BELOW    ############################


def check_split(input_ids):
    ''' 
    Splits data if there are more than 512 tokens, which is BERT's 
    limit context input
    '''
    # print("length of input_ids: "+str(len(input_ids)))
    # Keep track of answer splits
    answer_splits = []
    # Total number of splits due to the number of tokens and BERT limit of 512
    num_of_splits = 0
    # Number of tokens
    len_input = len(input_ids)

    # Determine number of splits required
    if (len_input > 512):
        if len_input % 512 == 0:
            num_of_splits = len_input / 512
        else:
            num_of_splits = int(len_input / 512) + 1

    # print("number of splits: "+str(num_of_splits))

    temp_ids = input_ids
    # Compute answer for all splits
    # for i in range(0, num_of_splits+1):
    for i in range(1):
        ind = (i+1)*512
        text_split_i = temp_ids[:ind]
        # print("ind: "+str(ind))
        # print("Computing answer for split i: " + str(text_split_i))
        answer_splits.append(split_answer(text_split_i))
        temp_ids = input_ids[ind:]
    # print("testing to see if rest still works")
    # print(split_answer(input_ids))

    return answer_splits


def answer_question(question, answer_text):
    '''
    Tokenizes the question and answer text
    '''

    input_ids = tokenizer.encode(question, answer_text)
    check_split(input_ids)


def split_answer(input_ids):
    '''
    Determines answer start and ending indexes for split_input
    '''
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

#####


def start_of_sentence(tokens, answer_start, answer_end):

    # Start with the first token and/or first letter in sentence
    if tokens[answer_start-1] == '.' or tokens[answer_start] == None:
        answer = tokens[answer_start]

    # Otherwise keep searching for the beginning of the sentence
    else:
        # ADD OR StATEMENT TOO??
        while tokens[answer_start].isalnum() == False:
            print(tokens[answer_start])
            answer_start -= 1

    # Find the last letter in the sentence
    # ADD OR StATEMENT TOO??
    if tokens[answer_end].isalnum() == False:
        while (tokens[answer_end] != '.' or answer_end+1 >= len(tokens)) and (len(tokens) > answer_end+1):
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

######################   TESTING     ###########################################


bert_abstract = "Purchased Content. When you purchase an item of content, your content will be stored in a digital locker and you may view it an unlimited number of times for during your Locker Period. The “Locker Period” will be for at least 5 years from the date of your purchase (subject to the restrictions described in the YouTube Paid Service Terms of Service). Each item of purchased content may have a different Locker Period and you agree to the Locker Period before you order it. Pausing, stopping, or rewinding purchased content will not extend the Locker Period.  As noted in the YouTube Paid Service Terms of Service, if an item of purchased content becomes unavailable during the five year period from the purchase date, you may request a refund. For purchased content: you may view one stream of each item at a time, you may view up to 3 streams of different items at a time, you may authorize up to 5 devices for offline playback of Locker Video Content at a time and to authorize additional devices, you must deauthorize one of those 5 devices, you may only authorize the same device three times in any 12 month period and de-authorize the same device twice in any 12 month period, you may only deauthorize a total of 2 devices for offline playback every 90 days, and you may only authorize 3 Google accounts on the same device. Stream and offline playback limitations for purchased content apply regardless of which Google product (e.g., Google Play Movies & TV or YouTube) you access the content from."
question = "Can I get a refund?"

answer_question(question, bert_abstract)

################################################################################

#########  SCRAPING  ###########################################################


# def tag_visible(element):
#     if element.parent.name in ['style', 'script', 'head', 'title', 'meta', '[document]']:
#         return False
#     if isinstance(element, Comment):
#         return False
#     return True


# def text_from_html(body):
#     soup = BeautifulSoup(body, 'html.parser')
#     texts = soup.findAll(text=True)
#     visible_texts = filter(tag_visible, texts)
#     return u" ".join(t.strip() for t in visible_texts)


# # youtube: https://www.youtube.com/t/usage_paycontent
# # tinder: https://policies.tinder.com/terms/us/en
# html = urllib.request.urlopen('https://policies.tinder.com/terms/us/en').read()
# print(text_from_html(html))

################################################################################

# bert_abstract = text_from_html(html)
# question = "Can I get a refund?"

# answer_question(question, bert_abstract)

################################################################################

print("hello world")
