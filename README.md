# image_to_text_to_audio

 # Image to Storyline Generator 

This Flask app uses BLIP and FLAN-T5 to:
- Generate a caption from an image
- Turn that caption into a 20-word creative story
- Convert the story to audio using gTTS

 Tech Stack
- Flask
- Hugging Face Transformers (`Salesforce/blip-image-captioning-base`, `google/flan-t5-base`)
- gTTS
- Pillow (for image handling)
