import os
import subprocess
import urllib.request
import telebot
import random

BOT_TOKEN = 'BOT_TOKEN'
bot = telebot.TeleBot(BOT_TOKEN)


@bot.message_handler(commands=['start', 'hello'])
def send_welcome(message):
    bot.reply_to(message, "Привет! Скажи что-нибудь в микрофон!")


def convert_oga_to_mp3(audio_url):
    urllib.request.urlretrieve(audio_url, 'audio.oga')
    subprocess.run(["ffmpeg", "-i", 'audio.oga', 'audio.mp3'])


def get_sex():
    sexes = ['мужчинa', 'женщина']
    with open('audio.mp3', 'rb') as data:
        answer = random.choice(sexes)
        probability = random.randint(1, 100)

    return answer, probability


@bot.message_handler(content_types=['voice'])
def telegram_bot(message, bot_token=BOT_TOKEN):
    file_info = bot.get_file(message.voice.file_id)
    audio_url = 'https://api.telegram.org/file/bot{}/{}'.format(bot_token, file_info.file_path)
    bot.send_message(message.chat.id, 'Скачиваю аудио ...')
    convert_oga_to_mp3(audio_url)
    bot.send_message(message.chat.id, 'Определяю пол ...')
    answer, probability = get_sex()
    bot.reply_to(message, 'Ваш пол - {} с вероятностью {}%'.format(answer, probability))

    os.remove("audio.mp3")
    os.remove("audio.oga")


bot.infinity_polling()
