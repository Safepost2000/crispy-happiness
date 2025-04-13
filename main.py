import os
import google.generativeai as genai
from dotenv import load_dotenv
from telegram import Update, InputFile, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import ApplicationBuilder, MessageHandler, CommandHandler, filters, ContextTypes

load_dotenv()

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
model = genai.GenerativeModel('gemini-pro-vision')
BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    keyboard = [[InlineKeyboardButton("Try sending a photo", callback_data='send_photo')]]
    reply_markup = InlineKeyboardMarkup(keyboard)
    await update.message.reply_text("Send a message or an image!", reply_markup=reply_markup)

async def button_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.callback_query.answer()
    await update.callback_query.message.reply_text("Just send me a photo now!")

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.message.from_user
    photo = update.message.photo
    voice = update.message.voice
    text = update.message.caption or update.message.text

    if voice:
        file = await context.bot.get_file(voice.file_id)
        file_path = f"{user.id}_voice.ogg"
        await file.download_to_drive(file_path)
        await update.message.reply_text("Voice received. I can't transcribe yet, but it's coming soon.")
        os.remove(file_path)
        return

    if photo:
        file = await context.bot.get_file(photo[-1].file_id)
        image_path = f"{user.id}_image.jpg"
        await file.download_to_drive(image_path)

        with open(image_path, "rb") as img:
            response = model.generate_content([text or "Describe this image", img])
        os.remove(image_path)
    else:
        response = model.generate_content(text)

    await update.message.reply_text(response.text)

def main():
    app = ApplicationBuilder().token(BOT_TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.TEXT | filters.PHOTO | filters.VOICE, handle_message))
    app.add_handler(MessageHandler(filters.COMMAND, start))
    app.add_handler(MessageHandler(filters.ALL & filters.StatusUpdate.CALLBACK_QUERY, button_handler))
    app.run_polling()

if __name__ == '__main__':
    main()
