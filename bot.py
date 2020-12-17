from sentence_transformers import SentenceTransformer
import numpy as np
from scipy.special import softmax
import telebot
import config

def cosine(u, v):
    return np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))


def find_best(sentence,
              model,
              dialogue_embeddings,
              dialogue_texts,
              song_embeddings,
              song_texts,
              k=10,
              use_dialogs=True):
    embedding = model.encode([sentence])[0]
    if use_dialogs:
        cos = cosine(dialogue_embeddings, embedding)
        idx = np.argmax(cos)
        if len(dialogue_texts[idx + 1].strip()) != 0 and len(dialogue_texts[idx].strip()) != 0:
            embedding = dialogue_embeddings[idx + 1]


model = SentenceTransformer('distilbert-base-nli-mean-tokens')

dialogue_texts = []
with open("dialogue.txt", "r") as f:
    i = 0
    for line in f:
        dialogue_texts += line.split('__eou__')

song_texts = []
song_file = {}
with open('data_final.tsv', 'r') as f:
    for line in f:
        data = line.split('\t')
        line, filename = data[1], data[3].strip()
        song_file[line] = filename
        if len(song_texts) == 0 or line != song_texts[-1]:
            song_texts.append(line)
            

song_embeddings = np.load('song_embeddings.npy')
dialogue_embeddings = np.load('dialogue_embeddings.npy')


    cos = cosine(song_embeddings, embedding)
    idx = np.argpartition(cos, -k)[-k:]
    p = softmax(cos[idx])
    idx = np.random.choice(idx, p=p)
    return song_texts[idx] if use_dialogs else song_texts[idx + 1]




bot = telebot.TeleBot(config.token)


@bot.message_handler(commands=["start"])
def start_game(message):
    bot.send_message(message.chat.id, "Hi! You may talk to me. Just write some message.")

@bot.message_handler(content_types=["text"])
def find_file_ids(message):
    
    text = find_best(message.text, model, dialogue_embeddings, dialogue_texts,
            song_embeddings, song_texts, k=1, use_dialogs=True)
    filename = song_file[text]
    print(filename)

    f = open('songs_final/' + filename, 'rb')
    msg = bot.send_voice(message.chat.id, f, None)
#    bot.reply_to(message, text)
    
if __name__ == '__main__':
    bot.infinity_polling()
