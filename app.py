from flask import Flask, render_template, request, send_from_directory
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration, pipeline
from gtts import gTTS
import os
import uuid

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/audio'

# Load models once
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
caption_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
story_gen = pipeline("text2text-generation", model="google/flan-t5-base")

def generate_caption(image):
    inputs = processor(image, return_tensors="pt")
    out = caption_model.generate(**inputs)
    return processor.decode(out[0], skip_special_tokens=True)

def generate_story(caption):
    prompt = f"Write a very short and detailed maginative story in 20 words based on this: '{caption}'"
    story = story_gen(prompt, max_length=30, do_sample=True)[0]['generated_text']
    return story
def clean_caption(caption):
    words = caption.split()
    seen = set()
    result = []
    for word in words:
        if word not in seen or word in ['a', 'in', 'the']:
            result.append(word)
            seen.add(word)
    return ' '.join(result)
def text_to_speech(text):
    tts = gTTS(text)
    filename = f"story_{uuid.uuid4().hex}.mp3"
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    tts.save(filepath)
    return filename

@app.route('/', methods=['GET', 'POST'])
def index():
    story = None
    audio_file = None

    if request.method == 'POST':
        file = request.files['image']
        if file:
            image = Image.open(file.stream)
            image.save('static/temp_image.jpg')
            caption = clean_caption(generate_caption(image))
            story = generate_story(caption)
            audio_file = text_to_speech(story)

    return render_template('index.html', story=story, audio_file=audio_file)

@app.route('/audio/<filename>')
def audio(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    os.makedirs('static/audio', exist_ok=True)
    app.run(debug=True)
