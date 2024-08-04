import telegram
from telegram.ext import ApplicationBuilder
import asyncio
from datetime import datetime

from utils.constant import TOKEN, USER_ID

app = telegram.Bot(TOKEN)

async def send_photo(photo_path: str) -> None:
    async with app:
        await app.send_photo(USER_ID, photo_path)

async def send_message(text) -> None:
    async with app:
        await app.send_message(USER_ID, text)

async def handle_door(photo_path: str, text: str) -> None:
    async with app:
        await send_photo(photo_path)
        await send_message(text)
    
