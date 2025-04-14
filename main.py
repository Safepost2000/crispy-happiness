# bot.py
# Complete code with fixes, error handling, and enhancements.

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
load_dotenv()  # Load environment variables from .env file for local development

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# --- Gemini Model Configuration ---
# Model for understanding text and images (multimodal input)
VISION_MODEL_NAME = "gemini-1.5-flash"

# Model for generating images.
# !!! IMPORTANT !!! Replace this with the actual, available model name from Google AI
# that supports API-based image generation. Check Google AI Studio / Vertex AI documentation.
# Examples: "models/gemini-1.5-pro-latest", "models/imagen-2..." (if using Vertex AI)
GENERATION_MODEL_NAME = os.getenv("GENERATION_MODEL_NAME", "models/gemini-1.5-flash-latest") # Verify this model name!

# Configure Google Generative AI
if GOOGLE_API_KEY:
    try:
        genai.configure(api_key=GOOGLE_API_KEY)
        logger_gemini = logging.getLogger("google.generativeai")
        logger_gemini.setLevel(logging.WARNING) # Optional: Reduce Gemini logging noise
    except Exception as e:
        logging.critical(f"Failed to configure Google Generative AI: {e}")
        # Consider exiting if configuration fails
        # exit("Google AI Configuration Failed.")
else:
    logging.warning("GOOGLE_API_KEY environment variable not found.")

# Initialize models (handle potential errors if key/model is invalid)
vision_model = None
generation_model = None
try:
    if GOOGLE_API_KEY: # Only attempt if API key is present
        vision_model = genai.GenerativeModel(VISION_MODEL_NAME)
        logging.info(f"Successfully initialized vision model: {VISION_MODEL_NAME}")

        # Only initialize generation model if the name is configured and seems plausible
        if GENERATION_MODEL_NAME:
             try:
                 generation_model = genai.GenerativeModel(GENERATION_MODEL_NAME)
                 logging.info(f"Successfully initialized generation model: {GENERATION_MODEL_NAME}")
             except Exception as e:
                 logging.error(f"Failed to initialize generation model '{GENERATION_MODEL_NAME}': {e}. Image generation disabled.")
                 generation_model = None # Ensure it's None if init fails
        else:
             logging.warning("Image Generation model name not configured. Generation command disabled.")
             generation_model = None

except Exception as e:
    logging.error(f"Failed to initialize one or more Gemini models: {e}")
    # vision_model and generation_model might be None here already


# Enable logging
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)
# Set higher logging level for httpx and other noisy libraries
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("telegram.vendor.ptb_urllib3.urllib3").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)

# --- Helper Functions ---
async def download_image(file_id: str, context: ContextTypes.DEFAULT_TYPE) -> bytes | None:
    """Downloads an image from Telegram into memory."""
    bot = context.bot
    try:
        file = await bot.get_file(file_id)
        # Download file content into a BytesIO object
        file_stream = io.BytesIO()
        await file.download_to_memory(file_stream)
        file_stream.seek(0)
        logger.debug(f"Successfully downloaded image with file_id {file_id}")
        return file_stream.read() # Return the bytes
    except TelegramError as e:
        logger.error(f"Telegram error downloading image with file_id {file_id}: {e}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error downloading image: {e}", exc_info=True)
        return None

# --- Command Handlers ---
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Sends a welcome message when the /start command is issued."""
    user_name = update.effective_user.first_name
    start_message = (
        f"Hello {user_name}! 👋 I'm your Gemini bot.\n\n"
        "➡️ Send me text, or text with an image, and I'll respond using Gemini.\n"
    )
    if generation_model: # Only mention generation if the model is available
         start_message += f"➡️ Use `/generate_image [your prompt]` to create an image (powered by {GENERATION_MODEL_NAME})."
    else:
        start_message += "Image generation is currently disabled."

    await update.message.reply_text(start_message)


async def generate_image_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Generates an image based on the user's prompt using the specified Gemini model."""
    # Check if the generation model was initialized successfully
    if not generation_model:
        await update.message.reply_text("Sorry, the image generation feature is not available or not configured correctly.")
        return

    if not context.args:
        await update.message.reply_text("Please provide a description for the image after the command.\nUsage: `/generate_image a futuristic cityscape at sunset`")
        return

    prompt = " ".join(context.args)
    user = update.effective_user
    chat_id = update.effective_chat.id
    logger.info(f"User {user.id} in chat {chat_id} requested image generation with prompt: '{prompt}'")

    # Let the user know processing has started
    try:
        processing_message = await update.message.reply_text(f"⏳ Generating image for prompt: '{prompt}'...", disable_notification=True)
    except TelegramError as e:
        logger.error(f"Failed to send 'generating' message in chat {chat_id}: {e}")
        processing_message = None # Continue without the status message if sending failed

    try:
        # --- Image Generation API Call ---
        # IMPORTANT: Verify the API call structure for your specific GENERATION_MODEL_NAME.
        # This assumes generate_content works and returns image data in response.parts.
        generation_full_prompt = f"Generate an image depicting: {prompt}" # Example prompt structure
        response = await generation_model.generate_content_async(
             generation_full_prompt,
             # generation_config={"candidate_count": 1}, # Example config
             # safety_settings=... # Add safety settings if needed
        )

        # --- Process Generation Response ---
        generated_image_bytes = None
        if response.parts:
            for part in response.parts:
                 if part.mime_type and part.mime_type.startswith("image/"):
                     # Assuming data is in part.blob - VERIFY THIS for your model
                     if hasattr(part, 'blob') and isinstance(part.blob, bytes):
                         generated_image_bytes = part.blob
                         logger.info(f"Image ({part.mime_type}) generated successfully (via blob) for user {user.id}.")
                         break
                     # Add checks for other potential data attributes if needed (e.g., part.data)

        # Clean up the "Generating..." message if it was sent
        if processing_message:
            try:
                await context.bot.delete_message(chat_id=processing_message.chat_id, message_id=processing_message.message_id)
            except TelegramError as e:
                logger.warning(f"Failed to delete 'generating' message in chat {chat_id}: {e}")


        if generated_image_bytes:
            image_file = io.BytesIO(generated_image_bytes)
            # Try to determine a reasonable file name extension
            extension = part.mime_type.split('/')[-1] if part.mime_type else 'png'
            image_file.name = f"generated_image.{extension}"

            try:
                await update.message.reply_photo(photo=InputFile(image_file), caption=f"✨ Here's the image for: '{prompt}'")
            except TelegramError as te:
                logger.error(f"Telegram error sending generated photo for user {user.id}: {te}")
                await update.message.reply_text(f"I generated the image, but failed to send it. Error: {te}. Please try again.")
        else:
            # Check for reasons why generation might have failed without throwing an exception
            error_text = "Sorry, I couldn't generate an image. No image data received in the response."
            if hasattr(response, 'prompt_feedback') and response.prompt_feedback:
                 error_text = f"Sorry, generation failed. Reason: {response.prompt_feedback}"
                 logger.warning(f"Image generation failed for user {user.id} due to prompt feedback: {response.prompt_feedback}")
            elif hasattr(response, 'text') and response.text: # Sometimes errors are returned as text
                error_text = f"Sorry, I received text instead of an image: {response.text}"
                logger.warning(f"Image generation for user {user.id} returned text: {response.text}")
            else:
                logger.warning(f"Image generation failed for user {user.id}. No image data and no specific feedback found in response: {response}")

            await update.message.reply_text(error_text)

    except (genai.types.BlockedPromptException, genai.types.StopCandidateException) as safety_exception:
        logger.warning(f"Image generation safety block for user {user.id}. Prompt: '{prompt}'. Reason: {safety_exception}")
        if processing_message: # Try to delete status message on safety block
             try:
                 await context.bot.delete_message(chat_id=processing_message.chat_id, message_id=processing_message.message_id)
             except TelegramError: pass
        await update.message.reply_text("Sorry, your request was blocked due to safety reasons or content policies. Please modify your prompt.")

    except Exception as e:
        logger.error(f"Error during image generation processing for user {user.id}: {e}", exc_info=True)
        if processing_message: # Try to delete status message on general error
            try:
                await context.bot.delete_message(chat_id=processing_message.chat_id, message_id=processing_message.message_id)
            except TelegramError: pass
        await update.message.reply_text("😥 An unexpected error occurred while trying to generate the image. Please try again later.")


# --- Message Handler ---
async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handles text messages and messages with photos for multimodal input."""
    # Check if the vision model was initialized successfully
    if not vision_model:
        await update.message.reply_text("Sorry, the bot is not fully configured to process messages at the moment.")
        return

    message = update.message
    user = update.effective_user
    chat_id = update.effective_chat.id
    user_input_text = message.text or message.caption # Use caption if text is empty (common for images)
    user_image: Image.Image | None = None # Type hint for PIL Image
    image_bytes: bytes | None = None
    processing_msg = None # To hold the status message object

    logger.info(f"Received message from user {user.id} in chat {chat_id}. Text/Caption: '{user_input_text}', Photo: {message.photo is not None}")

    # 1. Handle Photo Input
    if message.photo:
        try:
            processing_msg = await update.message.reply_text("⏳ Processing image...", disable_notification=True)
        except TelegramError as e:
             logger.error(f"Failed to send 'processing image' message in chat {chat_id}: {e}")
             processing_msg = None

        photo_file_id = message.photo[-1].file_id # Get the largest resolution photo
        image_bytes = await download_image(photo_file_id, context)

        if image_bytes:
            try:
                # Open image using Pillow to send to Gemini
                user_image = Image.open(io.BytesIO(image_bytes))
                # You might want to resize large images here if needed to save tokens/time
                # user_image.thumbnail((max_width, max_height))
                logger.info(f"Image downloaded and opened successfully for user {user.id}.")
            except Exception as e:
                logger.error(f"Error opening downloaded image bytes for user {user.id}: {e}")
                if processing_msg: await processing_msg.edit_text("Sorry, I couldn't process the image file format.")
                else: await update.message.reply_text("Sorry, I couldn't process the image file format.")
                return # Stop processing if image fails to open
        else:
            if processing_msg: await processing_msg.edit_text("Sorry, I failed to download the image from Telegram.")
            else: await update.message.reply_text("Sorry, I failed to download the image from Telegram.")
            return # Stop processing if download fails

        # Clean up the "Processing image..." message if it exists and image processing was successful so far
        if processing_msg:
            try:
                await context.bot.delete_message(chat_id=processing_msg.chat_id, message_id=processing_msg.message_id)
                processing_msg = None # Reset after deletion
            except TelegramError as e:
                logger.warning(f"Failed to delete 'processing image' message in chat {chat_id}: {e}")


    # 2. Prepare Content for Gemini API
    gemini_input_parts = []
    if user_input_text:
        gemini_input_parts.append(user_input_text.strip())
    if user_image:
        # Append the PIL Image object directly (google-generativeai handles this)
        gemini_input_parts.append(user_image)

    # Check if there's anything to send (e.g., user sent only an image that failed to process)
    if not gemini_input_parts:
        logger.warning(f"No text or valid image content to process for user {user.id}.")
        # Optionally inform user if only an image was sent but failed processing earlier
        # await update.message.reply_text("Please send text or ensure the image is valid.")
        return

    # 3. Call Gemini API - show "Thinking..." status
    try:
        # Avoid sending duplicate status message if 'processing image' failed to delete
        if not processing_msg:
             processing_msg = await update.message.reply_text("🧠 Thinking...", disable_notification=True)
    except TelegramError as e:
        logger.error(f"Failed to send 'thinking' message in chat {chat_id}: {e}")
        processing_msg = None

    try:
        # Use the vision model for multimodal input
        response = await vision_model.generate_content_async(
            gemini_input_parts,
            # Add safety_settings or generation_config if needed
            # generation_config={"candidate_count": 1},
            # safety_settings=...
        )

        # Clean up the "Thinking..." message
        if processing_msg:
            try:
                await context.bot.delete_message(chat_id=processing_msg.chat_id, message_id=processing_msg.message_id)
            except TelegramError as e:
                logger.warning(f"Failed to delete 'thinking' message in chat {chat_id}: {e}")


        # 4. Send Gemini's Response
        if hasattr(response, 'text') and response.text:
             response_text = response.text
             logger.info(f"Received Gemini text response for user {user.id}. Length: {len(response_text)}")
             # Send text response (chunking for long messages)
             max_length = constants.MessageLimit.TEXT_LENGTH
             for i in range(0, len(response_text), max_length):
                 chunk = response_text[i:i+max_length]
                 try:
                     await update.message.reply_text(chunk, parse_mode=constants.ParseMode.MARKDOWN)
                 except TelegramError as te:
                     # If markdown fails, try sending as plain text
                     if "can't parse entities" in str(te).lower():
                         logger.warning(f"Markdown parsing failed for chunk for user {user.id}. Sending as plain text. Error: {te}")
                         try:
                             await update.message.reply_text(chunk)
                         except TelegramError as te_plain:
                             logger.error(f"Failed to send plain text chunk for user {user.id}: {te_plain}")
                     else:
                         logger.error(f"Telegram error sending chunk for user {user.id}: {te}")
        else:
            # Handle cases where response might be blocked or empty
            fallback_text = "I received your message, but I didn't get a text response back from the AI."
            if hasattr(response, 'prompt_feedback') and response.prompt_feedback:
                 fallback_text += f"\nReason: {response.prompt_feedback}"
                 logger.warning(f"Gemini response blocked for user {user.id}. Feedback: {response.prompt_feedback}")
            else:
                 logger.warning(f"Gemini response for user {user.id} was empty. Response object: {response}")
            await update.message.reply_text(fallback_text)


    except (genai.types.BlockedPromptException, genai.types.StopCandidateException) as safety_exception:
        logger.warning(f"Gemini processing safety block for user {user.id}. Reason: {safety_exception}")
        if processing_msg: # Try to delete status message
             try: await context.bot.delete_message(chat_id=processing_msg.chat_id, message_id=processing_msg.message_id)
             except TelegramError: pass
        await update.message.reply_text("Sorry, your request was blocked due to safety reasons or content policies.")

    except Exception as e:
        logger.error(f"Error calling Gemini API or processing response for user {user.id}: {e}", exc_info=True)
        if processing_msg: # Try to delete status message
             try: await context.bot.delete_message(chat_id=processing_msg.chat_id, message_id=processing_msg.message_id)
             except TelegramError: pass
        await update.message.reply_text("😥 An unexpected error occurred while processing your request with the AI. Please try again later.")


# --- Error Handler ---
async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Log Errors caused by Updates."""
    logger.error("Exception while handling an update:", exc_info=context.error)

    # Specifically handle the Conflict error to avoid spamming logs but still be aware
    if isinstance(context.error, Conflict):
        logger.warning("Conflict error detected: %s. Ensure only ONE bot instance is running!", context.error)
        # In a production scenario, you might want to exit or notify someone if this persists.
        # For Render, it might happen briefly during deploys, so just logging might be okay.

    # You could add more specific error handling here if needed
    # For example, handling specific Telegram API errors gracefully

    # Optional: Inform user about generic errors (use with caution)
    # try:
    #     if isinstance(update, Update) and update.effective_message:
    #         # Avoid sending error message for Conflict errors as they are internal issues
    #         if not isinstance(context.error, Conflict):
    #              await update.effective_message.reply_text("Sorry, an unexpected error occurred processing your request.")
    # except Exception as e:
    #     logger.error(f"Failed to send error notification message to user: {e}")


# --- Main Execution Function ---
async def amain() -> None:
    """Initializes and starts the Telegram bot."""
    if not TELEGRAM_BOT_TOKEN:
        logger.critical("TELEGRAM_BOT_TOKEN environment variable not set! Exiting.")
        return
    if not GOOGLE_API_KEY:
        logger.warning("GOOGLE_API_KEY environment variable not set! AI features may fail.")
    if not vision_model and not generation_model:
        logger.critical("Neither Vision nor Generation models initialized successfully. Bot may not function. Exiting.")
        return
    elif not vision_model:
         logger.warning("Vision model failed to initialize. Text/Image processing commands will not work.")
    elif not generation_model:
         logger.warning("Generation model not initialized or failed to initialize. /generate_image command is disabled.")

    logger.info("--- Starting Bot Application ---")
    persistence = None # Add persistence here if needed later (e.g., PicklePersistence)
    application = (
        Application.builder()
        .token(TELEGRAM_BOT_TOKEN)
        .persistence(persistence)
        .build()
    )

    # --- Add Error Handler FIRST ---
    application.add_error_handler(error_handler)

    # --- Register Handlers ---
    application.add_handler(CommandHandler("start", start))
    if generation_model: # Only register if model is available
        application.add_handler(CommandHandler("generate_image", generate_image_command))
        logger.info("Registered /generate_image command handler.")
    else:
        logger.info("Skipping /generate_image handler registration (model unavailable).")

    # Message handler for text (excluding commands) OR photos (with potential captions)
    application.add_handler(MessageHandler(
        (filters.TEXT & ~filters.COMMAND) | filters.PHOTO, handle_message
    ))
    logger.info("Registered message handler for text and photos.")


    # --- Initialize, Clean Up Potential Conflicts, and Run ---
    try:
        logger.info("Initializing application...")
        await application.initialize()

        # Crucial: Drop pending updates and ensure no webhook is set.
        logger.info("Dropping pending Telegram updates and ensuring no webhook is set...")
        await application.bot.delete_webhook(drop_pending_updates=True)
        logger.info("Webhook check/cleanup complete.")

        logger.info("Bot initialized successfully. Starting polling...")
        await application.run_polling(
            allowed_updates=Update.ALL_TYPES, # Process all update types relevant to handlers
            drop_pending_updates=True, # Drop updates that arrived while bot was offline
            # timeout=30 # Optional: Increase polling timeout
        )

    except Conflict as e:
         # Catch conflict specifically on startup phase
         logger.error(f"Conflict detected during startup or polling init: {e}. Another instance is likely running. Shutting down this instance.")
         # No need to call shutdown() here, as run_polling likely didn't start fully

    except Exception as e:
         logger.error(f"An error occurred during bot startup or polling initialization: {e}", exc_info=True)
         # No need to call shutdown() here, as run_polling likely didn't start fully

    finally:
        # Graceful shutdown if polling was started and then stopped/encountered error
        if application.running:
            logger.info("Shutting down application polling...")
            await application.shutdown()
        logger.info("--- Bot Application Shutdown ---")


# --- Script Entry Point ---
if __name__ == "__main__":
    logger.info(f"Starting script execution at {__name__}")
    # Run the async main function using asyncio
    try:
        asyncio.run(amain())
    except KeyboardInterrupt:
        logger.info("Bot stopped manually via KeyboardInterrupt.")
    except Exception as e:
        # Catch any unexpected errors during asyncio.run or initial setup phase
        logger.critical(f"Application failed to run due to an unhandled exception: {e}", exc_info=True)
    finally:
        logger.info("Script execution finished.")

