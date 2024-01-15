import os
import subprocess
import urllib.request
import telebot
import random
import pickle

import librosa
import numpy as np
import torch
import torch.nn.functional as F

from model import Model, preprocess_audio
from linear_model import extract_features


mode = "cnn"
BOT_TOKEN = 'BOT_TOKEN'
bot = telebot.TeleBot(BOT_TOKEN)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Model()
model.load_state_dict(torch.load("model.pth", map_location=device))
with open("linear_model.pkl", "rb") as f:
    lr = pickle.load(f)

@bot.message_handler(commands=['start', 'hello'])
def send_welcome(message):
    bot.reply_to(message, "Привет! Скажи что-нибудь в микрофон!")


def get_sex():
    sexes = ['мужчинa', 'женщина']
    audio, _ = librosa.load("audio.oga", sr=48000)
    if mode == "cnn":
        data = preprocess_audio(audio).unsqueeze(0)
        logit = model(data.to(device)).squeeze(0)
        proba = F.sigmoid(logit).cpu().detach().item()
    elif mode == "linear":
        data = extract_features(audio)[np.newaxis]
        proba = lr.predict_proba(data)[0][1]
    return sexes[proba <  0.5], max(proba, 1 - proba)


@bot.message_handler(content_types=['voice'])
def telegram_bot(message, bot_token=BOT_TOKEN):
    file_info = bot.get_file(message.voice.file_id)
    audio_url = 'https://api.telegram.org/file/bot{}/{}'.format(bot_token, file_info.file_path)
    bot.send_message(message.chat.id, 'Скачиваю аудио ...')
    urllib.request.urlretrieve(audio_url, 'audio.oga')
    bot.send_message(message.chat.id, 'Определяю пол ...')
    answer, probability = get_sex()
    bot.reply_to(message, f'Ваш пол - {answer} с вероятностью {probability:.2%}')

    os.remove("audio.oga")


print("Бот запущен")
bot.infinity_polling()
