telegram_token = '7336469994:AAGvqqk5Ca8d5g9FiiNNM0bSjyUDnvK0-DY'

user_id = 2109292535

import telegram
import asyncio

from datetime import datetime

bot = telegram.Bot(telegram_token)

async def info():

    print(await bot.get_me())

    print('testing step =====> ....... 1/1 ')

    pic = 'https://bitcoin.org/img/icons/opengraph.png'
    await bot.send_photo(user_id, pic)

    text = 'Door Open' + datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    await bot.send_message(user_id, text)

    print('testing step =====> ....... 1/2 ')


asyncio.run(info())

print('testing step =====> ....... 1/3 ')

print('testing step =====> ....... 1/4 ')