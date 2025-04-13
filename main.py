import os
import logging
from io import BytesIO
import requests
from dotenv import load_dotenv
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, ContextTypes, filters
import google.generativeai as genai
from PIL import Image

# Enable logging
logging.basicConfig(format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Get tokens from environment variables
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Configure the Gemini API
genai.configure(api_key=GEMINI_API_KEY)

# Set up the model
model = genai.GenerativeModel('gemini-2.0-flash-exp-image-generation')

# Store user conversations
user_conversations = {}

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Send a message when the command /start is issued."""
    await update.message.reply_text(
        "Hi! I'm a Gemini AI bot. Send me text or images, and I'll respond to them."
    )
    
    # Initialize a new conversation for the user
    user_id = update.effective_user.id
    user_conversations[user_id] = model.start_chat(history=[])

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Send a message when the command /help is issued."""
    await update.message.reply_text("Send me text or images to chat with Gemini AI!")

async def reset(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Reset the conversation history."""
    user_id = update.effective_user.id
    user_conversations[user_id] = model.start_chat(history=[])
    await update.message.reply_text("Conversation history has been reset!")

async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle text messages."""
    user_id = update.effective_user.id
    
    # Initialize conversation if it doesn't exist
    if user_id not in user_conversations:
        user_conversations[user_id] = model.start_chat(history=[])
    
    try:
        # Get response from Gemini
        response = user_conversations[user_id].send_message(update.message.text)
        await update.message.reply_text(response.text)
    except Exception as e:
        logger.error(f"Error: {e}")
        await update.message.reply_text(f"Error: {str(e)}")

async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle photos."""
    try:
        # Get the photo file
        photo_file = await context.bot.get_file(update.message.photo[-1].file_id)
        photo_bytes = await photo_file.download_as_bytearray()
        
        # Get the caption if it exists
        caption = update.message.caption if update.message.caption else "Describe this image."
        
        # Generate response from Gemini
        response = model.generate_content([
            caption,
            {"mime_type": "image/jpeg", "data": photo_bytes}
        ])
        
        await update.message.reply_text(response.text)
    except Exception as e:
        logger.error(f"Error: {e}")
        await update.message.reply_text(f"Error processing image: {str(e)}")

def main() -> None:
    """Start the bot."""
    # Create the Application
    application = Application.builder().token(TELEGRAM_TOKEN).build()

    # Add handlers
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(CommandHandler("reset", reset))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))
    application.add_handler(MessageHandler(filters.PHOTO, handle_photo))

    # Run the bot
    application.run_polling()

if __name__ == "__main__":
    main()
