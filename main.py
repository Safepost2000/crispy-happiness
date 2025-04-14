# main.py (or bot.py - ensure filename matches Render start command)
# Complete code with simplified main loop handling

import logging
import os
import io
import requests
import asyncio # Import asyncio
from dotenv import load_dotenv
from PIL import Image
import google.generativeai as genai
from telegram import Update, InputMediaPhoto, InputFile, constants
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    filters,
    ContextTypes,
)
from telegram.error import TelegramError, Conflict # Import Conflict error

# --- Configuration ---
load_dotenv()
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
VISION_MODEL_NAME = "gemini-1.5-flash"
# !!! IMPORTANT: Verify/Update this model name !!!
GENERATION_MODEL_NAME = os.getenv("GENERATION_MODEL_NAME", "models/gemini-1.5-flash-latest")

# --- Logging Setup ---
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("telegram.vendor.ptb_urllib3.urllib3").setLevel(logging.WARNING)
logging.getLogger("google.generativeai").setLevel(logging.WARNING) # Optional: Reduce Gemini logging
logger = logging.getLogger(__name__)

# --- Gemini Configuration & Initialization ---
vision_model = None
generation_model = None
if GOOGLE_API_KEY:
    try:
        genai.configure(api_key=GOOGLE_API_KEY)
        vision_model = genai.GenerativeModel(VISION_MODEL_NAME)
        logger.info(f"Successfully initialized vision model: {VISION_MODEL_NAME}")
        if GENERATION_MODEL_NAME:
            try:
                generation_model = genai.GenerativeModel(GENERATION_MODEL_NAME)
                logger.info(f"Successfully initialized generation model: {GENERATION_MODEL_NAME}")
            except Exception as e:
                logger.error(f"Failed to initialize generation model '{GENERATION_MODEL_NAME}': {e}. Image generation disabled.")
                generation_model = None
        else:
            logging.warning("Image Generation model name not configured. Generation command disabled.")
            generation_model = None
    except Exception as e:
        logging.error(f"Failed to configure Google Generative AI or initialize models: {e}")
else:
    logging.warning("GOOGLE_API_KEY environment variable not found.")

# --- Helper Functions ---
async def download_image(file_id: str, context: ContextTypes.DEFAULT_TYPE) -> bytes | None:
    """Downloads an image from Telegram into memory."""
    bot = context.bot
    try:
        file = await bot.get_file(file_id)
        file_stream = io.BytesIO()
        await file.download_to_memory(file_stream)
        file_stream.seek(0)
        logger.debug(f"Successfully downloaded image with file_id {file_id}")
        return file_stream.read()
    except TelegramError as e:
        logger.error(f"Telegram error downloading image with file_id {file_id}: {e}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error downloading image: {e}", exc_info=True)
        return None

# --- Command Handlers ---
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Sends a welcome message."""
    user_name = update.effective_user.first_name
    start_message = (
        f"Hello {user_name}! 👋 I'm your Gemini bot.\n\n"
        "➡️ Send me text, or text with an image, and I'll respond using Gemini.\n"
    )
    if generation_model:
         start_message += f"➡️ Use `/generate_image [your prompt]` to create an image."
    else:
        start_message += "Image generation is currently disabled."
    await update.message.reply_text(start_message)

async def generate_image_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Generates an image based on the user's prompt."""
    if not generation_model:
        await update.message.reply_text("Sorry, the image generation feature is not available.")
        return
    if not context.args:
        await update.message.reply_text("Usage: `/generate_image [your prompt]`")
        return

    prompt = " ".join(context.args)
    user = update.effective_user
    chat_id = update.effective_chat.id
    logger.info(f"User {user.id} in chat {chat_id} requested image generation: '{prompt}'")
    processing_message = None
    try:
        processing_message = await update.message.reply_text(f"⏳ Generating image for: '{prompt}'...", disable_notification=True)
    except TelegramError as e:
        logger.error(f"Failed to send 'generating' message in chat {chat_id}: {e}")

    try:
        generation_full_prompt = f"Generate an image depicting: {prompt}"
        response = await generation_model.generate_content_async(generation_full_prompt)

        generated_image_bytes = None
        image_mime_type = "image/png" # Default
        if response.parts:
            for part in response.parts:
                 if part.mime_type and part.mime_type.startswith("image/"):
                     if hasattr(part, 'blob') and isinstance(part.blob, bytes):
                         generated_image_bytes = part.blob
                         image_mime_type = part.mime_type
                         logger.info(f"Image ({image_mime_type}) generated successfully for user {user.id}.")
                         break

        if processing_message:
            try: await context.bot.delete_message(chat_id=processing_message.chat_id, message_id=processing_message.message_id)
            except TelegramError: pass # Ignore if deletion fails

        if generated_image_bytes:
            image_file = io.BytesIO(generated_image_bytes)
            extension = image_mime_type.split('/')[-1] if image_mime_type else 'png'
            image_file.name = f"generated_image.{extension}"
            try:
                await update.message.reply_photo(photo=InputFile(image_file), caption=f"✨ Image for: '{prompt}'")
            except TelegramError as te:
                logger.error(f"Telegram error sending generated photo for user {user.id}: {te}")
                await update.message.reply_text(f"I generated the image, but failed to send it. Error: {te}")
        else:
            error_text = "Sorry, couldn't generate image. No image data received."
            if hasattr(response, 'prompt_feedback') and response.prompt_feedback:
                 error_text = f"Sorry, generation failed. Reason: {response.prompt_feedback}"
                 logger.warning(f"Image generation failed (prompt feedback): {response.prompt_feedback}")
            elif hasattr(response, 'text') and response.text:
                error_text = f"Sorry, received text instead of image: {response.text}"
                logger.warning(f"Image generation returned text: {response.text}")
            else:
                logger.warning(f"Image generation failed. No image data/feedback. Response: {response}")
            await update.message.reply_text(error_text)

    except (genai.types.BlockedPromptException, genai.types.StopCandidateException) as safety_exception:
        logger.warning(f"Image generation safety block for user {user.id}. Reason: {safety_exception}")
        if processing_message:
             try: await context.bot.delete_message(chat_id=processing_message.chat_id, message_id=processing_message.message_id)
             except TelegramError: pass
        await update.message.reply_text("Request blocked due to safety/content policies.")
    except Exception as e:
        logger.error(f"Error during image generation for user {user.id}: {e}", exc_info=True)
        if processing_message:
            try: await context.bot.delete_message(chat_id=processing_message.chat_id, message_id=processing_message.message_id)
            except TelegramError: pass
        await update.message.reply_text("😥 Unexpected error generating image.")

# --- Message Handler ---
async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handles text messages and messages with photos for multimodal input."""
    if not vision_model:
        await update.message.reply_text("Bot AI component not ready.")
        return

    message = update.message
    user = update.effective_user
    chat_id = update.effective_chat.id
    user_input_text = message.text or message.caption
    user_image: Image.Image | None = None
    processing_msg = None

    logger.info(f"Msg from User:{user.id} Chat:{chat_id} Text:'{user_input_text}' Photo:{message.photo is not None}")

    if message.photo:
        try: processing_msg = await update.message.reply_text("⏳ Processing image...", disable_notification=True)
        except TelegramError as e: logger.error(f"Failed to send 'processing image' msg: {e}")

        photo_file_id = message.photo[-1].file_id
        image_bytes = await download_image(photo_file_id, context)

        if image_bytes:
            try: user_image = Image.open(io.BytesIO(image_bytes))
            except Exception as e:
                logger.error(f"Error opening image bytes: {e}")
                msg = "Sorry, couldn't process image file format."
                if processing_msg: await processing_msg.edit_text(msg)
                else: await update.message.reply_text(msg)
                return
        else:
            msg = "Sorry, failed to download image."
            if processing_msg: await processing_msg.edit_text(msg)
            else: await update.message.reply_text(msg)
            return

        if processing_msg: # Delete "Processing..." msg now if successful so far
            try:
                await context.bot.delete_message(chat_id=processing_msg.chat_id, message_id=processing_msg.message_id)
                processing_msg = None
            except TelegramError: pass

    gemini_input_parts = []
    if user_input_text: gemini_input_parts.append(user_input_text.strip())
    if user_image: gemini_input_parts.append(user_image)

    if not gemini_input_parts:
        logger.warning(f"No text/valid image content for user {user.id}.")
        return

    if not processing_msg: # Show thinking message if not already showing (or deleted)
        try: processing_msg = await update.message.reply_text("🧠 Thinking...", disable_notification=True)
        except TelegramError as e: logger.error(f"Failed to send 'thinking' msg: {e}")

    try:
        response = await vision_model.generate_content_async(gemini_input_parts)

        if processing_msg: # Delete "Thinking..." message
            try: await context.bot.delete_message(chat_id=processing_msg.chat_id, message_id=processing_msg.message_id)
            except TelegramError: pass

        if hasattr(response, 'text') and response.text:
             response_text = response.text
             logger.info(f"Gemini response length for user {user.id}: {len(response_text)}")
             max_length = constants.MessageLimit.TEXT_LENGTH
             for i in range(0, len(response_text), max_length):
                 chunk = response_text[i:i+max_length]
                 try: await update.message.reply_text(chunk, parse_mode=constants.ParseMode.MARKDOWN)
                 except TelegramError as te:
                     if "can't parse entities" in str(te).lower():
                         logger.warning(f"Markdown failed, sending plain text. Error: {te}")
                         try: await update.message.reply_text(chunk)
                         except TelegramError as te_plain: logger.error(f"Failed plain text send: {te_plain}")
                     else: logger.error(f"Telegram error sending chunk: {te}")
        else:
            fallback_text = "Received message, but no text response from AI."
            if hasattr(response, 'prompt_feedback') and response.prompt_feedback:
                 fallback_text += f"\nReason: {response.prompt_feedback}"
                 logger.warning(f"Gemini response blocked. Feedback: {response.prompt_feedback}")
            else: logger.warning(f"Gemini response empty. Response: {response}")
            await update.message.reply_text(fallback_text)

    except (genai.types.BlockedPromptException, genai.types.StopCandidateException) as safety_exception:
        logger.warning(f"Gemini processing safety block for user {user.id}. Reason: {safety_exception}")
        if processing_msg:
             try: await context.bot.delete_message(chat_id=processing_msg.chat_id, message_id=processing_msg.message_id)
             except TelegramError: pass
        await update.message.reply_text("Request blocked due to safety/content policies.")
    except Exception as e:
        logger.error(f"Error calling Gemini API or processing response: {e}", exc_info=True)
        if processing_msg:
             try: await context.bot.delete_message(chat_id=processing_msg.chat_id, message_id=processing_msg.message_id)
             except TelegramError: pass
        await update.message.reply_text("😥 Unexpected error processing request with AI.")

# --- Error Handler ---
async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Log Errors caused by Updates."""
    logger.error("Exception while handling an update:", exc_info=context.error)
    if isinstance(context.error, Conflict):
        logger.warning("Conflict error detected: %s. Ensure only ONE bot instance is running!", context.error)

# --- Main Execution Function ---
async def amain() -> None:
    """Initializes and starts the Telegram bot's polling."""
    if not TELEGRAM_BOT_TOKEN:
        logger.critical("TELEGRAM_BOT_TOKEN missing! Exiting.")
        return
    if not GOOGLE_API_KEY:
        logger.warning("GOOGLE_API_KEY missing! AI features may fail.")
    if not vision_model and not generation_model:
        logger.critical("Neither Vision nor Generation models initialized. Bot may not function. Exiting.")
        return

    logger.info("--- Starting Bot Application ---")
    application = Application.builder().token(TELEGRAM_BOT_TOKEN).build()

    application.add_error_handler(error_handler)

    # Register handlers
    application.add_handler(CommandHandler("start", start))
    if generation_model:
        application.add_handler(CommandHandler("generate_image", generate_image_command))
        logger.info("Registered /generate_image command.")
    application.add_handler(MessageHandler(
        (filters.TEXT & ~filters.COMMAND) | filters.PHOTO, handle_message
    ))
    logger.info("Registered message handler.")

    # Initialize and Run Polling (Let PTB manage the loop and shutdown)
    try:
        logger.info("Initializing application and starting polling...")
        # drop_pending_updates=True handles cleanup of old updates/webhook state
        await application.run_polling(
            allowed_updates=Update.ALL_TYPES,
            drop_pending_updates=True,
        )
    except Exception as e:
        # Catch errors during the initial setup of run_polling itself
        logger.error(f"An error occurred that stopped run_polling setup: {e}", exc_info=True)

# --- Script Entry Point ---
if __name__ == "__main__":
    # Ensure the filename below matches how you run it (main.py or bot.py)
    logger.info(f"Starting script execution (__main__)...")
    try:
        asyncio.run(amain())
    except KeyboardInterrupt:
        logger.info("Bot stopped manually via KeyboardInterrupt.")
    except Exception as e:
        logger.critical(f"Application failed to run: {e}", exc_info=True)
    finally:
        logger.info("--- Script execution finished ---")

