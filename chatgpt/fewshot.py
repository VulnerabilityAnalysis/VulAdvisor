import os.path

import pandas as pd
import tiktoken
from openai import OpenAI
import time
import random
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize

key = 'omitted' 
client = OpenAI(api_key=key)

def convert_string(s):
    words = s.split()

    if words[0].endswith('ing'):
        word = words[0]
        present_tense = nltk.stem.WordNetLemmatizer().lemmatize(word, 'v')
        words[0] = present_tense
    s = ' '.join(words)

    s = s[0].upper() + s[1:]

    return s

def tokenize_desc(text):
    text = text.replace('was', 'can be').replace('were', 'can be')
    text = text.replace("The vulnerability can be fixed by ", "")
    text = text.replace('"', "").replace("'", "").replace("`", "")
    sentences = sent_tokenize(text)
    words = sum([word_tokenize(s) for s in sentences], [])
    return convert_string(' '.join(words))

def get_token_count(message, model="gpt-3.5-turbo"):
    """Returns the number of tokens used by a list of messages."""
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")
    if model == "gpt-3.5-turbo":  # note: future models may deviate from this
        num_tokens =  4  # every message follows <im_start>{role/name}\n{content}<im_end>\n
        num_tokens += len(encoding.encode(message))
        return num_tokens
    else:
        raise NotImplementedError(f"""num_tokens_from_messages() is not presently implemented for model {model}.""")


def create_prompt_with_limit(train_data, test_function, max_tokens):
    """
    Create a prompt with 3 training examples and a test function, ensuring the total token count is within the limit.
    """
    for _ in range(1000):  # Try different combinations up to 100 times
        random.shuffle(train_data)
        selected_samples = train_data[:3]
        prompt = ""
        for func, suggestion in selected_samples:
            suggestion = tokenize_desc(suggestion)
            prompt += f"Input: {func}\nOutput: {suggestion}\n\n"
        prompt += f"Input: {test_function}\nOutput:"
        length = get_token_count(prompt)
        print(length)
        if length <= max_tokens:
            return prompt

    raise ValueError("Unable to create a prompt within the token limit with 3 examples.")

def query_chatgpt():
    min_cost = 0.5
    train_df = pd.read_csv('../../data/train.csv')
    train_data = list(train_df[['function', 'suggestion']].itertuples(index=False, name=None))

    # Iterate over the test set
    MAX_TOKENS = 4096
    df = pd.read_csv('../../data/test.csv')
    print(df.shape)
    with open('few-shot.out', 'r') as file:
        lines = [eval(line.split('\t')[0]) for line in file.readlines()]
    cursor = len(lines)

    with open('few-shot.out', 'a') as ret:
        for (i, item) in df.iterrows():
            # if i > 10:
            #     break
            if i <  cursor:
                continue
            print('=' * 50, i, '=' * 50)
            func = item['function']
            prompt = create_prompt_with_limit(train_data, func, MAX_TOKENS)
            while True:
                try:
                    completion = client.chat.completions.create(
                        temperature=0,
                        model="gpt-3.5-turbo",
                        messages=[{"role": "user", "content": prompt}]
                    )

                    # print(completion.choices[0].message)
                    break
                except Exception as e:
                    print(e)
                    time.sleep(60)

            reason = completion.choices[0].finish_reason
            if reason == 'stop':
                content = completion.choices[0].message.content
                print(content)
                ret.write(str(i)+'\t'+content.replace('\n', '')+'\n')
            else:
                print(reason)
                print(completion.choices[0].message.content)
            time.sleep(min_cost)

    # df.to_csv('')



if __name__ == '__main__':
    token_num = 0
    query_chatgpt()
