import os
import asyncio
import logging
import tempfile
from io import BytesIO
import base64
from dotenv import load_dotenv
from telegram import Update, Bot, InlineQueryResultArticle, InputTextMessageContent
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    ContextTypes,
    ConversationHandler,
    filters,
    InlineQueryHandler
)
import google.generativeai as genai
from PIL import Image
import requests
import uuid
import json
import time

# Enable logging
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Get tokens from environment variables
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Configure the Gemini API
genai.configure(api_key=GEMINI_API_KEY)

# Set up the models
TEXT_MODEL = 'gemini-2.0-pro'
VISION_MODEL = 'gemini-2.0-pro'

# Conversation states
CHATTING, SYSTEM_PROMPT = range(2)

# Store user conversations and settings
user_data = {}

# System instructions for the bot
DEFAULT_SYSTEM_INSTRUCTION = (
    "You are an advanced AI assistant powered by Google's Gemini. "
    "You are helpful, concise, accurate, and friendly. "
    "You can process images and text to provide helpful information. "
    "Always respond in a conversational manner and try to be as helpful as possible."
)

def get_user_chat(user_id):
    """Get or create a chat for a user."""
    if user_id not in user_data:
        user_data[user_id] = {
            "chat": genai.GenerativeModel(TEXT_MODEL).start_chat(
                history=[],
                system_instruction=DEFAULT_SYSTEM_INSTRUCTION
            ),
            "system_instruction": DEFAULT_SYSTEM_INSTRUCTION,
            "settings": {
                "temperature": 0.7,
                "top_p": 0.95,
                "top_k": 64,
                "max_output_tokens": 8192,
            }
        }
    return user_data[user_id]["chat"]

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Send a message when the command /start is issued."""
    user = update.effective_user
    user_id = user.id
    
    # Initialize user data if needed
    if user_id not in user_data:
        get_user_chat(user_id)
        
    welcome_message = (
        f"👋 Hello {user.mention_html()}! I'm an advanced Gemini AI assistant.\n\n"
        f"I can understand both text and images, and maintain a conversation with you.\n\n"
        f"🔹 Send me text messages to chat\n"
        f"🔹 Send images to get descriptions\n"
        f"🔹 Send images with captions to ask questions about them\n\n"
        f"Type /help to see all available commands."
    )
    
    await update.message.reply_html(welcome_message)
    return CHATTING

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Send a message when the command /help is issued."""
    help_text = (
        "🤖 *Gemini Multimodal Chatbot* 🤖\n\n"
        "*Commands:*\n"
        "/start - Start or restart the bot\n"
        "/help - Show this help message\n"
        "/reset - Reset your conversation history\n"
        "/system - Set a custom system instruction\n"
        "/settings - View current model settings\n"
        "/temperature - Set temperature (0.0-1.0)\n\n"
        "*Capabilities:*\n"
        "• Text conversations with memory\n"
        "• Image analysis and understanding\n"
        "• Combined image and text queries\n"
        "• Inline query support for quick responses\n\n"
        "*Tips:*\n"
        "• For the best image analysis, provide clear images\n"
        "• Add specific questions as captions with your images\n"
        "• Use /reset if the conversation gets off track\n\n"
        "Powered by Google's Gemini 2.0 Pro"
    )
    await update.message.reply_markdown(help_text)
    return CHATTING

async def reset(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Reset the conversation history."""
    user_id = update.effective_user.id
    
    # Get the current system instruction
    system_instruction = DEFAULT_SYSTEM_INSTRUCTION
    if user_id in user_data:
        system_instruction = user_data[user_id].get("system_instruction", DEFAULT_SYSTEM_INSTRUCTION)
    
    # Create a new chat with the same system instruction
    user_data[user_id] = {
        "chat": genai.GenerativeModel(TEXT_MODEL).start_chat(
            history=[],
            system_instruction=system_instruction
        ),
        "system_instruction": system_instruction,
        "settings": user_data.get(user_id, {}).get("settings", {
            "temperature": 0.7,
            "top_p": 0.95,
            "top_k": 64,
            "max_output_tokens": 8192,
        })
    }
    
    await update.message.reply_text("✅ Conversation history has been reset. Let's start fresh!")
    return CHATTING

async def start_system_prompt(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Start the process of setting a custom system instruction."""
    await update.message.reply_text(
        "Please enter a new system instruction. This will define how I behave in our conversation.\n\n"
        "Current instruction:\n"
        f"`{user_data.get(update.effective_user.id, {}).get('system_instruction', DEFAULT_SYSTEM_INSTRUCTION)}`\n\n"
        "Send /cancel to keep the current instruction."
    )
    return SYSTEM_PROMPT

async def set_system_prompt(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Set the custom system instruction."""
    user_id = update.effective_user.id
    new_instruction = update.message.text
    
    if new_instruction.lower() == '/cancel':
        await update.message.reply_text("System instruction update canceled.")
        return CHATTING
    
    # Update the system instruction and create a new chat
    if user_id not in user_data:
        get_user_chat(user_id)
        
    user_data[user_id]["system_instruction"] = new_instruction
    user_data[user_id]["chat"] = genai.GenerativeModel(TEXT_MODEL).start_chat(
        history=user_data[user_id]["chat"].history,
        system_instruction=new_instruction
    )
    
    await update.message.reply_text("✅ System instruction updated successfully!")
    return CHATTING

async def show_settings(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Show current model settings."""
    user_id = update.effective_user.id
    
    if user_id not in user_data:
        get_user_chat(user_id)
    
    settings = user_data[user_id]["settings"]
    
    settings_text = (
        "⚙️ *Current Model Settings* ⚙️\n\n"
        f"• Temperature: `{settings.get('temperature', 0.7)}`\n"
        f"• Top-P: `{settings.get('top_p', 0.95)}`\n"
        f"• Top-K: `{settings.get('top_k', 64)}`\n"
        f"• Max Output Tokens: `{settings.get('max_output_tokens', 8192)}`\n\n"
        "Use /temperature followed by a value between 0.0 and 1.0 to adjust temperature.\n"
        "Example: `/temperature 0.8`"
    )
    
    await update.message.reply_markdown(settings_text)
    return CHATTING

async def set_temperature(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Set temperature for the model."""
    user_id = update.effective_user.id
    
    if user_id not in user_data:
        get_user_chat(user_id)
    
    try:
        # Extract the temperature value from the command
        args = context.args
        if not args:
            raise ValueError("No temperature value provided")
        
        temp = float(args[0])
        if temp < 0.0 or temp > 1.0:
            raise ValueError("Temperature must be between 0.0 and 1.0")
        
        # Update the temperature setting
        user_data[user_id]["settings"]["temperature"] = temp
        
        await update.message.reply_text(f"✅ Temperature set to {temp}")
    except ValueError as e:
        await update.message.reply_text(
            f"❌ Error: {str(e)}\n"
            "Please use a value between 0.0 and 1.0.\n"
            "Example: `/temperature 0.8`"
        )
    
    return CHATTING

async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Handle text messages."""
    user_id = update.effective_user.id
    user_message = update.message.text
    chat = get_user_chat(user_id)
    settings = user_data[user_id]["settings"]
    
    # Send typing action
    await context.bot.send_chat_action(chat_id=update.effective_chat.id, action="typing")
    
    try:
        # Create a generation config from user settings
        generation_config = genai.types.GenerationConfig(
            temperature=settings.get("temperature", 0.7),
            top_p=settings.get("top_p", 0.95),
            top_k=settings.get("top_k", 64),
            max_output_tokens=settings.get("max_output_tokens", 8192),
        )
        
        # Get response from Gemini
        response = await asyncio.to_thread(
            chat.send_message, 
            user_message,
            generation_config=generation_config
        )
        
        # Send the response back to the user
        await update.message.reply_text(response.text)
        
    except Exception as e:
        logger.error(f"Error while processing text message: {e}")
        await update.message.reply_text(
            "Sorry, I encountered an error while processing your message.\n"
            f"Error details: {str(e)}"
        )
    
    return CHATTING

async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Handle photos with or without captions."""
    user_id = update.effective_user.id
    chat = get_user_chat(user_id)
    settings = user_data[user_id]["settings"]
    
    # Send typing action
    await context.bot.send_chat_action(chat_id=update.effective_chat.id, action="typing")
    
    try:
        # Get the photo file
        photo_file = await context.bot.get_file(update.message.photo[-1].file_id)
        image_url = photo_file.file_path
        
        # Download the image
        response = requests.get(image_url)
        image_content = response.content
        
        # Get the caption if it exists, otherwise use a default prompt
        caption = update.message.caption if update.message.caption else "Describe this image in detail."
        
        # Create a vision model with user settings
        vision_model = genai.GenerativeModel(
            VISION_MODEL,
            generation_config=genai.types.GenerationConfig(
                temperature=settings.get("temperature", 0.7),
                top_p=settings.get("top_p", 0.95),
                top_k=settings.get("top_k", 64),
                max_output_tokens=settings.get("max_output_tokens", 8192),
            )
        )
        
        # Create multimodal prompt
        multimodal_prompt = [
            caption,
            {
                "mime_type": "image/jpeg",
                "data": base64.b64encode(image_content).decode('utf-8')
            }
        ]
        
        # Send typing action again for longer processing
        await context.bot.send_chat_action(chat_id=update.effective_chat.id, action="typing")
        
        # Generate response
        vision_response = await asyncio.to_thread(
            vision_model.generate_content,
            multimodal_prompt
        )
        
        # Send the response
        await update.message.reply_text(vision_response.text)
        
        # Add the interaction to the conversation history
        chat.history.append({
            "role": "user",
            "parts": [{"text": f"[Image shared with caption: {caption}]"}]
        })
        
        chat.history.append({
            "role": "model",
            "parts": [{"text": vision_response.text}]
        })
        
    except Exception as e:
        logger.error(f"Error while processing photo: {e}")
        await update.message.reply_text(
            "Sorry, I encountered an error while processing your image.\n"
            f"Error details: {str(e)}"
        )
    
    return CHATTING

async def handle_voice(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Handle voice messages."""
    # Send typing action
    await context.bot.send_chat_action(chat_id=update.effective_chat.id, action="typing")
    
    await update.message.reply_text(
        "I received your voice message, but audio processing isn't fully implemented yet.\n\n"
        "For now, please type your questions or send images for the best experience."
    )
    
    return CHATTING

async def handle_video(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Handle video messages."""
    # Send typing action
    await context.bot.send_chat_action(chat_id=update.effective_chat.id, action="typing")
    
    await update.message.reply_text(
        "I received your video, but video processing isn't fully implemented yet.\n\n"
        "For now, please send still images or text messages for the best experience."
    )
    
    return CHATTING

async def handle_document(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Handle document messages."""
    # Send typing action
    await context.bot.send_chat_action(chat_id=update.effective_chat.id, action="typing")
    
    await update.message.reply_text(
        "I received your document, but document processing isn't fully implemented yet.\n\n"
        "For now, please share the content as text or images for the best experience."
    )
    
    return CHATTING

async def handle_inline_query(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle inline queries."""
    query = update.inline_query.text
    if not query:
        return
    
    try:
        # Create a new generative model for inline queries
        inline_model = genai.GenerativeModel(TEXT_MODEL)
        
        # Get response from Gemini
        response = await asyncio.to_thread(
            inline_model.generate_content,
            query
        )
        
        # Create an inline result
        results = [
            InlineQueryResultArticle(
                id=str(uuid.uuid4()),
                title="Gemini Response",
                description=response.text[:100] + "..." if len(response.text) > 100 else response.text,
                input_message_content=InputTextMessageContent(response.text)
            )
        ]
        
        # Answer the inline query
        await update.inline_query.answer(results)
        
    except Exception as e:
        logger.error(f"Error in inline query: {e}")
        # In case of error, return a message about the error
        results = [
            InlineQueryResultArticle(
                id=str(uuid.uuid4()),
                title="Error Processing Query",
                description=f"Error: {str(e)}",
                input_message_content=InputTextMessageContent(
                    "Sorry, I encountered an error while processing your query."
                )
            )
        ]
        await update.inline_query.answer(results)

async def cancel(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Cancel current operation and return to chatting."""
    await update.message.reply_text("Operation canceled. Let's continue our conversation.")
    return CHATTING

async def error_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle errors."""
    logger.error(f"Update {update} caused error {context.error}")
    
    # Send message to the user if possible
    if update and update.effective_message:
        await update.effective_message.reply_text(
            "Sorry, something went wrong while processing your request.\n"
            "Please try again later or reset the conversation with /reset."
        )

def main() -> None:
    """Start the bot."""
    # Create the Application
    application = Application.builder().token(TELEGRAM_TOKEN).build()

    # Create a conversation handler with states
    conv_handler = ConversationHandler(
        entry_points=[CommandHandler("start", start)],
        states={
            CHATTING: [
                CommandHandler("help", help_command),
                CommandHandler("reset", reset),
                CommandHandler("system", start_system_prompt),
                CommandHandler("settings", show_settings),
                CommandHandler("temperature", set_temperature),
                MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text),
                MessageHandler(filters.PHOTO, handle_photo),
                MessageHandler(filters.VOICE, handle_voice),
                MessageHandler(filters.VIDEO, handle_video),
                MessageHandler(filters.Document.ALL, handle_document),
            ],
            SYSTEM_PROMPT: [
                CommandHandler("cancel", cancel),
                MessageHandler(filters.TEXT & ~filters.COMMAND, set_system_prompt),
            ],
        },
        fallbacks=[CommandHandler("cancel", cancel)],
    )
    
    application.add_handler(conv_handler)
    
    # Add inline query handler
    application.add_handler(InlineQueryHandler(handle_inline_query))
    
    # Add standalone command handlers for users who haven't started the conversation
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("help", help_command))
    
    # Register error handler
    application.add_error_handler(error_handler)
    
    # Run the bot until the user presses Ctrl-C
    application.run_polling(allowed_updates=Update.ALL_TYPES)

if __name__ == "__main__":
    main()
