import os
import asyncio
import logging
import tempfile
from io import BytesIO
import base64
import uuid
import time
import requests
from dotenv import load_dotenv
from telegram import Update, Bot, InputFile
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    ContextTypes,
    ConversationHandler,
    filters
)
import google.generativeai as genai
from PIL import Image
import speech_recognition as sr
from gtts import gTTS

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

# Set up the models - using the specified model for all operations
MODEL_NAME = "gemini-2.0-flash-exp-image-generation"  # The newer model you specified

# Conversation states
CHATTING = 0

# Store user data
user_data = {}

# System instructions for the bot
DEFAULT_SYSTEM_INSTRUCTION = (
    "You are an advanced AI assistant powered by Google's Gemini. "
    "You are helpful, creative, and friendly. "
    "When appropriate, generate image descriptions enclosed in triple square brackets, like: "
    "[[[Generate an image of a sunset over mountains]]] "
    "For voice responses, indicate when something should be spoken in triple angle brackets, like: "
    "<<<This is the text that should be converted to speech>>> "
    "Respond to user inputs in a helpful and informative manner."
)

def get_user_chat(user_id):
    """Get or create a chat for a user."""
    if user_id not in user_data:
        user_data[user_id] = {
            "chat": genai.GenerativeModel(MODEL_NAME).start_chat(
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
        f"👋 Hello {user.mention_html()}! I'm an advanced Gemini multimodal assistant.\n\n"
        f"I can understand and generate:\n"
        f"🔹 Text messages\n"
        f"🔹 Images\n"
        f"🔹 Voice messages\n\n"
        f"Try sending me any of these formats and I'll respond accordingly!\n"
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
        "/settings - View current model settings\n"
        "/temperature - Set temperature (0.0-1.0)\n\n"
        "*Multimodal Capabilities:*\n"
        "• Send text messages for normal conversation\n"
        "• Send images for analysis or to guide responses\n"
        "• Send voice messages for speech-to-text conversion\n"
        "• Receive text, image, and voice responses\n\n"
        "*Tips:*\n"
        "• For image generation, the bot recognizes special patterns in responses\n"
        "• Voice responses are generated from text responses when appropriate\n"
        "• Use /reset if the conversation gets off track\n\n"
        f"Powered by Google's {MODEL_NAME} model"
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
        "chat": genai.GenerativeModel(MODEL_NAME).start_chat(
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

async def show_settings(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Show current model settings."""
    user_id = update.effective_user.id
    
    if user_id not in user_data:
        get_user_chat(user_id)
    
    settings = user_data[user_id]["settings"]
    
    settings_text = (
        "⚙️ *Current Model Settings* ⚙️\n\n"
        f"• Model: `{MODEL_NAME}`\n"
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
        
        # Process the response for multimodal outputs
        await process_and_send_response(update, context, response.text)
        
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
        photo_bytes = await photo_file.download_as_bytearray()
        
        # Get the caption if it exists, otherwise use a default prompt
        caption = update.message.caption if update.message.caption else "What do you see in this image? Provide a detailed description."
        
        # Create a generative model with user settings
        model = genai.GenerativeModel(
            MODEL_NAME,
            generation_config=genai.types.GenerationConfig(
                temperature=settings.get("temperature", 0.7),
                top_p=settings.get("top_p", 0.95),
                top_k=settings.get("top_k", 64),
                max_output_tokens=settings.get("max_output_tokens", 8192),
            )
        )
        
        # Create multimodal prompt
        response = await asyncio.to_thread(
            model.generate_content,
            [
                caption,
                {"mime_type": "image/jpeg", "data": photo_bytes}
            ]
        )
        
        # Process the response for multimodal outputs
        await process_and_send_response(update, context, response.text)
        
        # Add the interaction to the conversation history
        chat.history.append({
            "role": "user",
            "parts": [{"text": f"[Image shared with caption: {caption}]"}]
        })
        
        chat.history.append({
            "role": "model",
            "parts": [{"text": response.text}]
        })
        
    except Exception as e:
        logger.error(f"Error while processing photo: {e}")
        await update.message.reply_text(
            "Sorry, I encountered an error while processing your image.\n"
            f"Error details: {str(e)}"
        )
    
    return CHATTING

async def handle_voice(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Handle voice messages by converting speech to text and then processing the text."""
    user_id = update.effective_user.id
    chat = get_user_chat(user_id)
    
    # Send typing action
    await context.bot.send_chat_action(chat_id=update.effective_chat.id, action="typing")
    
    try:
        # Get the voice file
        voice_file = await context.bot.get_file(update.message.voice.file_id)
        
        # Create a temporary file to save the voice message
        with tempfile.NamedTemporaryFile(delete=False, suffix='.ogg') as temp_voice:
            voice_path = temp_voice.name
            await voice_file.download_to_drive(custom_path=voice_path)
        
        # Convert to WAV for speech recognition
        wav_path = voice_path.replace('.ogg', '.wav')
        os.system(f"ffmpeg -i {voice_path} {wav_path} -y")
        
        # Perform speech recognition
        recognizer = sr.Recognizer()
        with sr.AudioFile(wav_path) as source:
            audio_data = recognizer.record(source)
            text = recognizer.recognize_google(audio_data)
        
        # Clean up temporary files
        os.unlink(voice_path)
        os.unlink(wav_path)
        
        # Inform the user their speech was recognized
        await update.message.reply_text(f"🎤 I heard: \"{text}\"\n\nProcessing your request...")
        
        # Process the transcribed text with Gemini
        generation_config = genai.types.GenerationConfig(
            temperature=user_data[user_id]["settings"].get("temperature", 0.7),
            top_p=user_data[user_id]["settings"].get("top_p", 0.95),
            top_k=user_data[user_id]["settings"].get("top_k", 64),
            max_output_tokens=user_data[user_id]["settings"].get("max_output_tokens", 8192),
        )
        
        # Get response from Gemini
        response = await asyncio.to_thread(
            chat.send_message, 
            text,
            generation_config=generation_config
        )
        
        # Process the response for multimodal outputs
        await process_and_send_response(update, context, response.text)
        
    except sr.UnknownValueError:
        await update.message.reply_text("Sorry, I couldn't understand the audio. Please try speaking more clearly.")
    except sr.RequestError:
        await update.message.reply_text("Sorry, there was an issue with the speech recognition service. Please try again later.")
    except Exception as e:
        logger.error(f"Error while processing voice message: {e}")
        await update.message.reply_text(
            "Sorry, I encountered an error while processing your voice message.\n"
            f"Error details: {str(e)}"
        )
    
    return CHATTING

async def process_and_send_response(update, context, response_text):
    """Process the response text to handle various response types (text, image, voice)."""
    chat_id = update.effective_chat.id
    
    # Check for image generation triggers
    image_prompts = extract_image_prompts(response_text)
    response_without_image_prompts = remove_image_prompts(response_text)
    
    # Check for voice response triggers
    voice_text, text_only = extract_voice_text(response_without_image_prompts)
    
    # First send the text response
    if text_only:
        await context.bot.send_message(chat_id=chat_id, text=text_only)
    
    # Then generate and send any images
    for image_prompt in image_prompts:
        await context.bot.send_chat_action(chat_id=chat_id, action="upload_photo")
        try:
            image_data = await generate_image(image_prompt)
            if image_data:
                with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as temp_img:
                    temp_img.write(image_data)
                    temp_img_path = temp_img.name
                
                # Send the generated image
                with open(temp_img_path, 'rb') as img:
                    await context.bot.send_photo(
                        chat_id=chat_id, 
                        photo=InputFile(img),
                        caption=f"Generated image based on: {image_prompt[:100]}..."
                    )
                
                # Clean up
                os.unlink(temp_img_path)
            else:
                await context.bot.send_message(
                    chat_id=chat_id,
                    text=f"I tried to generate an image for '{image_prompt}' but couldn't create it successfully."
                )
        except Exception as e:
            logger.error(f"Error generating image: {e}")
            await context.bot.send_message(
                chat_id=chat_id,
                text=f"I couldn't generate the requested image due to an error: {str(e)}"
            )
    
    # Finally, generate and send any voice responses
    if voice_text:
        await context.bot.send_chat_action(chat_id=chat_id, action="record_audio")
        try:
            voice_data = await generate_voice(voice_text)
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as temp_voice:
                temp_voice.write(voice_data)
                temp_voice_path = temp_voice.name
            
            # Send the generated voice message
            with open(temp_voice_path, 'rb') as voice:
                await context.bot.send_voice(
                    chat_id=chat_id,
                    voice=InputFile(voice)
                )
            
            # Clean up
            os.unlink(temp_voice_path)
        except Exception as e:
            logger.error(f"Error generating voice: {e}")
            await context.bot.send_message(
                chat_id=chat_id,
                text=f"I couldn't generate the voice response due to an error: {str(e)}"
            )

def extract_image_prompts(text):
    """Extract image generation prompts from text."""
    import re
    # Look for prompts enclosed in [[[prompt]]] format
    image_prompts = re.findall(r'\[\[\[(.*?)\]\]\]', text)
    return image_prompts

def remove_image_prompts(text):
    """Remove image prompts from text."""
    import re
    return re.sub(r'\[\[\[(.*?)\]\]\]', '', text)

def extract_voice_text(text):
    """Extract text that should be converted to speech."""
    import re
    voice_texts = re.findall(r'<<<(.*?)>>>', text)
    # Join all voice texts with spaces
    voice_text = ' '.join(voice_texts) if voice_texts else ""
    
    # Remove the voice markers from the text
    text_only = re.sub(r'<<<(.*?)>>>', '', text)
    
    return voice_text, text_only

async def generate_image(prompt):
    """Generate an image based on a text prompt.
    
    This function uses the Gemini image generation capability if available,
    or falls back to a third-party image generation service.
    """
    try:
        # Placeholder for actual image generation
        # In a real implementation, this would call the Gemini API or another image generation service
        
        # For demonstration, we'll use a placeholder image generation service
        # This code should be replaced with actual implementation
        
        # Simulate image generation with a delay
        await asyncio.sleep(2)
        
        # For testing, return a simple gradient image
        from PIL import Image, ImageDraw
        
        # Create a gradient image as a placeholder
        img = Image.new('RGB', (512, 512), color='white')
        draw = ImageDraw.Draw(img)
        
        # Draw a simple gradient
        for y in range(512):
            for x in range(512):
                r = int(255 * x / 512)
                g = int(255 * y / 512)
                b = int(128)
                draw.point((x, y), fill=(r, g, b))
        
        # Add some text to the image
        draw.text((20, 20), f"Image from prompt: {prompt[:50]}...", fill=(255, 255, 255))
        
        # Convert to bytes
        img_byte_array = BytesIO()
        img.save(img_byte_array, format='JPEG')
        return img_byte_array.getvalue()
        
    except Exception as e:
        logger.error(f"Image generation error: {e}")
        return None

async def generate_voice(text):
    """Generate voice audio from text using gTTS."""
    try:
        tts = gTTS(text=text, lang='en', slow=False)
        
        # Save to a BytesIO object
        mp3_fp = BytesIO()
        tts.write_to_fp(mp3_fp)
        mp3_fp.seek(0)
        
        return mp3_fp.read()
    except Exception as e:
        logger.error(f"Voice generation error: {e}")
        raise

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

    # Create a conversation handler
    conv_handler = ConversationHandler(
        entry_points=[CommandHandler("start", start)],
        states={
            CHATTING: [
                CommandHandler("help", help_command),
                CommandHandler("reset", reset),
                CommandHandler("settings", show_settings),
                CommandHandler("temperature", set_temperature),
                MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text),
                MessageHandler(filters.PHOTO, handle_photo),
                MessageHandler(filters.VOICE, handle_voice),
            ],
        },
        fallbacks=[CommandHandler("start", start)],
    )
    
    application.add_handler(conv_handler)
    
    # Add standalone command handlers for users who haven't started the conversation
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("help", help_command))
    
    # Register error handler
    application.add_error_handler(error_handler)
    
    # Run the bot until the user presses Ctrl-C
    application.run_polling(allowed_updates=Update.ALL_TYPES)

if __name__ == "__main__":
    main()
