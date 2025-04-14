import logging
import os
import io
import requests
from dotenv import load_dotenv
from PIL import Image
import google.generativeai as genai
from telegram import Update, InputMediaPhoto, InputFile, constants
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
from telegram.error import TelegramError

# --- Configuration ---
load_dotenv()  # Load environment variables from .env file for local development

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# --- Gemini Model Configuration ---
# Model for understanding text and images (multimodal input)
# 'gemini-1.5-flash' is a good choice for this.
VISION_MODEL_NAME = "gemini-1.5-flash"

# Model for generating images.
# !!! IMPORTANT !!! Replace this with the actual, available model name from Google AI
# that supports API-based image generation. The user-provided name might be experimental.
# Check Google AI Studio / Vertex AI documentation.
# If using Vertex AI Imagen, the API call structure will be different.
GENERATION_MODEL_NAME = os.getenv("GENERATION_MODEL_NAME", "models/gemini-1.5-flash-latest") # Placeholder - Verify correct model!
# Example placeholder: GENERATION_MODEL_NAME = "models/imagen-2..." (if using Vertex)

# Configure Google Generative AI
if GOOGLE_API_KEY:
    genai.configure(api_key=GOOGLE_API_KEY)
else:
    logging.warning("GOOGLE_API_KEY not found in environment variables.")
    # Exit or handle gracefully if API key is missing
    # exit("API key missing.") # Uncomment to exit if API key is crucial

# Initialize models (handle potential errors if key/model is invalid)
try:
    vision_model = genai.GenerativeModel(VISION_MODEL_NAME)
    # Only initialize generation model if the name seems plausible or configured
    if GENERATION_MODEL_NAME and GOOGLE_API_KEY:
         generation_model = genai.GenerativeModel(GENERATION_MODEL_NAME)
         logging.info(f"Using generation model: {GENERATION_MODEL_NAME}")
    else:
         generation_model = None
         logging.warning("Image Generation model not configured or API key missing.")

except Exception as e:
    logging.error(f"Failed to initialize Gemini models: {e}")
    # Consider exiting or disabling features if models fail to load
    vision_model = None
    generation_model = None


# Enable logging
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)
# Set higher logging level for httpx to avoid excessive noise
logging.getLogger("httpx").setLevel(logging.WARNING)
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
        return file_stream.read() # Return the bytes
    except TelegramError as e:
        logger.error(f"Telegram error downloading image with file_id {file_id}: {e}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error downloading image: {e}")
        return None

# --- Command Handlers ---
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Sends a welcome message when the /start command is issued."""
    user_name = update.effective_user.first_name
    await update.message.reply_text(
        f"Hello {user_name}! I'm your Gemini bot.\n\n"
        "➡️ Send me text, or text with an image, and I'll respond.\n"
        f"➡️ Use `/generate_image [your prompt]` to create an image (if enabled)."
    )

async def generate_image_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Generates an image based on the user's prompt using the specified Gemini model."""
    if not generation_model:
        await update.message.reply_text("Sorry, the image generation feature is not available or not configured correctly.")
        return

    if not context.args:
        await update.message.reply_text("Please provide a description for the image after the command.\nUsage: `/generate_image a cat wearing a wizard hat`")
        return

    prompt = " ".join(context.args)
    user = update.effective_user
    logger.info(f"User {user.id} requested image generation with prompt: '{prompt}'")
    # Let the user know processing has started
    processing_message = await update.message.reply_text(f"⏳ Generating image for prompt: '{prompt}'...", disable_notification=True)

    try:
        # --- Image Generation API Call ---
        # IMPORTANT: This is a potential structure. The actual API call might differ.
        # You might need specific parameters, prompt formats, or even a different library
        # (like google-cloud-aiplatform for Vertex AI's Imagen).
        # Check the documentation for your specific GENERATION_MODEL_NAME.

        # Assuming generate_content can handle image generation prompts:
        response = await generation_model.generate_content_async(
             f"Generate an image depicting: {prompt}" # Example prompt structure
             # Add generation_config parameters if needed (e.g., number of candidates)
        )

        # --- Process Generation Response ---
        # Adapt this based on the ACTUAL response structure from your model.
        generated_image_bytes = None
        if response.parts:
            for part in response.parts:
                 # Check if the part contains image data (adjust mime types if needed)
                 if part.mime_type and part.mime_type.startswith("image/"):
                     # Assuming the image data is directly in 'part.blob' which might be specific to vision models
                     # It could be in 'part.data' or require specific handling. Verify this!
                     if hasattr(part, 'blob') and isinstance(part.blob, bytes):
                         generated_image_bytes = part.blob
                         logger.info(f"Image generated successfully (via blob). Mime type: {part.mime_type}")
                         break
                     # Add other potential checks based on API docs
                     # elif hasattr(part, 'data') and isinstance(part.data, bytes):
                     #    generated_image_bytes = part.data
                     #    logger.info(f"Image generated successfully (via data). Mime type: {part.mime_type}")
                     #    break

        # Clean up the "Generating..." message
        await context.bot.delete_message(chat_id=processing_message.chat_id, message_id=processing_message.message_id)

        if generated_image_bytes:
            image_file = io.BytesIO(generated_image_bytes)
            image_file.name = "generated_image.png" # Adjust extension based on mime_type if possible
            try:
                await update.message.reply_photo(photo=InputFile(image_file), caption=f"✨ Here's the image for: '{prompt}'")
            except TelegramError as te:
                logger.error(f"Telegram error sending generated photo: {te}")
                await update.message.reply_text(f"I generated the image, but failed to send it due to a Telegram error: {te}")
        else:
            # Check if response.text contains an explanation or error
            error_text = "Sorry, I couldn't generate an image. No image data received."
            if hasattr(response, 'text') and response.text:
                 error_text = f"Sorry, I couldn't generate an image. Response: {response.text}"
            elif response.prompt_feedback:
                 error_text = f"Sorry, generation failed due to: {response.prompt_feedback}"

            logger.warning(f"Image generation failed or no image data found. Response: {response}")
            await update.message.reply_text(error_text)

    except genai.types.BlockedPromptException as bpe:
        logger.warning(f"Image generation prompt blocked for user {user.id}. Reason: {bpe}")
        await context.bot.delete_message(chat_id=processing_message.chat_id, message_id=processing_message.message_id)
        await update.message.reply_text("Sorry, your prompt was blocked by safety filters. Please try a different prompt.")
    except genai.types.StopCandidateException as sce:
         logger.warning(f"Image generation stopped for user {user.id}. Reason: {sce}")
         await context.bot.delete_message(chat_id=processing_message.chat_id, message_id=processing_message.message_id)
         await update.message.reply_text("Sorry, the image generation was stopped, possibly due to content policies. Please try again or adjust your prompt.")
    except Exception as e:
        logger.error(f"Error during image generation for user {user.id}: {e}", exc_info=True)
        try:
            await context.bot.delete_message(chat_id=processing_message.chat_id, message_id=processing_message.message_id)
        except TelegramError:
            pass # Ignore if deleting the message fails
        await update.message.reply_text(f"😥 An unexpected error occurred while trying to generate the image. Please try again later.")


# --- Message Handler ---
async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handles text messages and messages with photos."""
    if not vision_model:
        await update.message.reply_text("Sorry, the bot is not properly configured to process messages.")
        return

    message = update.message
    user_input_text = message.text or message.caption # Use caption if text is empty (for images)
    user_image = None
    image_bytes = None
    user = update.effective_user

    logger.info(f"Received message from user {user.id}. Text/Caption: '{user_input_text}', Photo: {message.photo is not None}")

    # 1. Handle Photo Input
    if message.photo:
        # Inform user that image is being processed
        processing_msg = await update.message.reply_text("⏳ Processing image...", disable_notification=True)
        photo_file_id = message.photo[-1].file_id # Get the largest resolution photo
        image_bytes = await download_image(photo_file_id, context)

        if image_bytes:
            try:
                # Open image using Pillow to send to Gemini
                user_image = Image.open(io.BytesIO(image_bytes))
                logger.info(f"Image downloaded and opened successfully for user {user.id}.")
            except Exception as e:
                logger.error(f"Error opening downloaded image bytes: {e}")
                await processing_msg.edit_text("Sorry, I couldn't process the image file you sent.")
                return
        else:
            await processing_msg.edit_text("Sorry, I failed to download the image.")
            return
        # Clean up the "Processing..." message after download attempt
        try:
            await context.bot.delete_message(chat_id=processing_msg.chat_id, message_id=processing_msg.message_id)
        except TelegramError:
            pass # Ignore if deletion fails

    # 2. Prepare Content for Gemini
    gemini_input_parts = []
    if user_input_text:
        gemini_input_parts.append(user_input_text)
    if user_image:
        # Append the PIL Image object directly
        gemini_input_parts.append(user_image)

    # Check if there's anything to send
    if not gemini_input_parts:
        logger.warning(f"No text or valid image to process for user {user.id}.")
        # Optionally send a message if only an image *failed* to process but no text was sent
        # await update.message.reply_text("Please send text or ensure the image is valid.")
        return

    # 3. Call Gemini API
    processing_message = await update.message.reply_text("🧠 Thinking...", disable_notification=True)
    try:
        # Use the vision model for multimodal input
        response = await vision_model.generate_content_async(
            gemini_input_parts,
            # Add safety_settings or generation_config if needed
        )

        # Clean up the "Thinking..." message
        await context.bot.delete_message(chat_id=processing_message.chat_id, message_id=processing_message.message_id)

        # 4. Send Gemini's Response
        if response.text:
             # Send text response (consider chunking for long messages)
             max_length = constants.MessageLimit.TEXT_LENGTH
             for i in range(0, len(response.text), max_length):
                 chunk = response.text[i:i+max_length]
                 await update.message.reply_text(chunk, parse_mode=constants.ParseMode.MARKDOWN)
        else:
            # Handle cases where response might be blocked or empty
             logger.warning(f"Gemini response for user {user.id} was empty or blocked. Feedback: {response.prompt_feedback}")
             fallback_text = "I received your message, but I didn't get a text response back."
             if response.prompt_feedback:
                 fallback_text += f"\nReason: {response.prompt_feedback}"
             await update.message.reply_text(fallback_text)


    except genai.types.BlockedPromptException as bpe:
        logger.warning(f"Gemini request blocked for user {user.id}. Reason: {bpe}")
        await context.bot.delete_message(chat_id=processing_message.chat_id, message_id=processing_message.message_id)
        await update.message.reply_text("Sorry, your message was blocked by safety filters.")
    except genai.types.StopCandidateException as sce:
        logger.warning(f"Gemini response stopped for user {user.id}. Reason: {sce}")
        await context.bot.delete_message(chat_id=processing_message.chat_id, message_id=processing_message.message_id)
        await update.message.reply_text("Sorry, the response generation was stopped, possibly due to content policies.")
    except Exception as e:
        logger.error(f"Error calling Gemini API for user {user.id}: {e}", exc_info=True)
        try:
            await context.bot.delete_message(chat_id=processing_message.chat_id, message_id=processing_message.message_id)
        except TelegramError:
            pass
        await update.message.reply_text("😥 An unexpected error occurred while processing your request. Please try again later.")


# --- Main Execution ---
def main() -> None:
    """Starts the Telegram bot."""
    if not TELEGRAM_BOT_TOKEN:
        logger.critical("TELEGRAM_BOT_TOKEN environment variable not set! Exiting.")
        return
    if not GOOGLE_API_KEY:
        # Log warning but allow continuation if only vision is maybe needed or key set later
        logger.warning("GOOGLE_API_KEY environment variable not set! Some features might fail.")
    if not vision_model and not generation_model:
        logger.critical("Both Vision and Generation models failed to initialize. Exiting.")
        return
    elif not vision_model:
         logger.warning("Vision model failed to initialize. Text/Image processing will not work.")
    elif not generation_model:
         logger.warning("Generation model not initialized. /generate_image command will not work.")


    logger.info("Starting bot application...")
    application = Application.builder().token(TELEGRAM_BOT_TOKEN).build()

    # Register command handlers
    application.add_handler(CommandHandler("start", start))
    if generation_model: # Only add handler if model is available
        application.add_handler(CommandHandler("generate_image", generate_image_command))
    else:
        logger.info("Skipping /generate_image handler registration as model is unavailable.")

    # Register message handler for text (excluding commands) OR photos (with potential captions)
    application.add_handler(MessageHandler(
        (filters.TEXT & ~filters.COMMAND) | filters.PHOTO, handle_message
    ))

    # Start the Bot using polling
    logger.info("Bot started successfully. Running polling...")
    application.run_polling(allowed_updates=Update.ALL_TYPES) # Process all update types


if __name__ == "__main__":
    main()
