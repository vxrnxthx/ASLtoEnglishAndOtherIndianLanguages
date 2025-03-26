import mediapipe as mp
import cv2
import numpy as np
from gtts import gTTS
import os
import pygame
import tempfile
from cnn import Model, DataGatherer
from Auto_Correct_SpellChecker import Auto_Correct
from GUI import GUI
from tkinter import *
from tkinter import ttk
from PIL import ImageTk, Image
from googletrans import Translator

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
cap = None
translator = Translator()

# Initialize pygame mixer for playing audio
pygame.mixer.init()

# Pre-trained saved model with 99% accuracy
classifier = Model.load_classifier('model_trained.h5')


def text_to_speech(text, language='en'):
    """
    Perform text-to-speech for given text using gTTS.

    Args:
        text (str): Text to be converted to speech
        language (str, optional): Language code. Defaults to 'en'.
    """
    if not text or not text.strip():
        return

    try:
        # Create a temporary file to save the audio
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as temp_audio:
            # Generate speech
            tts = gTTS(text=text, lang=language)
            tts.save(temp_audio.name)

        # Play the audio
        pygame.mixer.music.load(temp_audio.name)
        pygame.mixer.music.play()

        # Wait for the audio to finish
        while pygame.mixer.music.get_busy():
            pygame.time.Clock().tick(10)

        # Remove the temporary file
        os.unlink(temp_audio.name)

    except Exception as e:
        print(f"Text-to-Speech error: {e}")


# Rest of the code remains the same as the previous implementation,
# with modifications to the pipe_cam function to use the new text_to_speech function

def draw_region(image, center):
    cropped_image = cv2.rectangle(image, (center[0] - 130, center[1] - 130),
                                  (center[0] + 130, center[1] + 130), (0, 0, 255), 2)
    return cropped_image[center[1] - 130:center[1] + 130, center[0] - 130:center[0] + 130], cropped_image


def start_gui(title, size):
    gui = GUI(title, size)

    # Configure root window with dark theme
    gui.root.configure(bg='#121212')

    # Create main container frames with dark theme
    left_frame = Frame(gui.root, bg='#121212')
    left_frame.grid(row=0, column=0, padx=10, pady=10, sticky='n')

    right_frame = Frame(gui.root, bg='#121212')
    right_frame.grid(row=0, column=1, padx=10, pady=10, sticky='n')

    # Create video frame in right frame
    gui_frame = Frame(right_frame, width=600, height=600, bg='#1E1E1E')
    gui_frame.pack(pady=10)
    vid_label = Label(gui_frame, bg='#1E1E1E')
    vid_label.pack()

    return gui, vid_label, left_frame


def exit_app(gui, cap):
    gui.root.destroy()
    cap.release()
    pygame.mixer.quit()


def update_frame(image, vid_label):
    image_fromarray = Image.fromarray(image)
    imgtk = ImageTk.PhotoImage(image=image_fromarray)
    vid_label.imgtk = imgtk
    vid_label.config(image=imgtk)


def get_threshold(label_entrybox):
    value = label_entrybox.get('1.0', END)
    try:
        return float(value)
    except:
        return 0.95


def get_char(gesture):
    classes = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L',
               'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V',
               'W', 'X', 'Y', 'Z', 'del', 'nothing', 'space']
    return Model.predict(classes, classifier, gesture)


def translate_text(sentence_box, translated_box, lang_combo):
    text = sentence_box.get('1.0', 'end-1c')
    if text:
        try:
            translated = translator.translate(text, dest=lang_combo.get())
            translated_box.delete('1.0', 'end')
            translated_box.insert('end', translated.text)
        except Exception as e:
            translated_box.delete('1.0', 'end')
            translated_box.insert('end', f"Translation error: {str(e)}")


def add_letter(curr_char, word_box, current_char_box):
    if curr_char and curr_char != 'nothing':
        word_box.insert('end', curr_char.lower())


def add_space(word_box, sentence_box, corrected_word_box):
    current_word = word_box.get('1.0', 'end-1c')
    if current_word:
        corrected = Auto_Correct(current_word)
        sentence_box.insert('end', corrected + " ")
        corrected_word_box.delete('1.0', 'end')
        corrected_word_box.insert('end', corrected)
        word_box.delete('1.0', 'end')


def delete_char(word_box):
    content = word_box.get('1.0', 'end-1c')
    if content:
        word_box.delete('1.0', 'end')
        word_box.insert('end', content[:-1])


def frame_video_stream(names, curr_char, prev_char, *args):
    kwargs = dict(zip(names, args))
    threshold = get_threshold(kwargs['th_box'])

    success, frame = cap.read()
    frame = cv2.flip(frame, 1)
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    update_frame(image, kwargs['vid_label'])

    image.flags.writeable = False
    results = kwargs['hands'].process(image)

    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    image_height, image_width, _ = image.shape

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            x = [landmark.x for landmark in hand_landmarks.landmark]
            y = [landmark.y for landmark in hand_landmarks.landmark]

            center = np.array([np.mean(x) * image_width, np.mean(y) * image_height]).astype('int32')
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            cropped_img, full_img = draw_region(image, center)

            update_frame(full_img, kwargs['vid_label'])

            try:
                gray = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2GRAY)
                gray = DataGatherer().edge_detection(gray)

                curr_char, pred = get_char(gray)
                char = cv2.putText(full_img, curr_char, (center[0] - 135, center[1] - 135),
                                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
                char_prob = cv2.putText(full_img, '{0:.2f}'.format(np.max(pred)),
                                        (center[0] + 60, center[1] - 135), cv2.FONT_HERSHEY_SIMPLEX,
                                        1, (0, 0, 255), 2, cv2.LINE_AA)

                update_frame(full_img, kwargs['vid_label'])

                if np.max(pred) > threshold:
                    kwargs['cc_box'].delete('1.0', 'end')
                    kwargs['cc_box'].insert('end', curr_char)
                    global current_predicted_char
                    current_predicted_char = curr_char

            except:
                pass

    kwargs['vid_label'].after(1, frame_video_stream, names, curr_char, prev_char, *args)


def pipe_cam(gui, vid_label, left_frame):
    curr_char = None
    prev_char = None
    threshold = float(0.95)
    global current_predicted_char
    current_predicted_char = None

    float_formatter = "{:.5f}".format
    np.set_printoptions(formatter={'float_kind': float_formatter})

    global cap
    cap = cv2.VideoCapture(0)

    # Dark theme style configurations
    bg_color = '#121212'
    fg_color = 'white'
    entry_bg = '#1E1E1E'
    button_bg = '#2C2C2C'
    button_fg = 'white'

    # Create labels and entry boxes in left frame
    labels = ['threshold', 'current char', 'original word', 'corrected word', 'sentence']

    # Create entry boxes container
    entry_frame = Frame(left_frame, bg=bg_color)
    entry_frame.pack(pady=5)

    entryboxes = {}
    for i, label in enumerate(labels):
        Label(entry_frame, text=label, bg=bg_color, fg=fg_color).grid(row=i, column=0, pady=2, sticky='w')
        entry = Text(entry_frame, height=1 if label != 'sentence' else 3, width=20,
                     bg=entry_bg, fg=fg_color, insertbackground='white')
        entry.grid(row=i, column=1, pady=2, padx=5)
        entryboxes[f'{label}_entrybox'] = entry

    entryboxes['threshold_entrybox'].insert('end', threshold)

    # Create translation frame
    trans_frame = Frame(left_frame, bg=bg_color)
    trans_frame.pack(pady=5)

    # Create translation boxes for each language
    translation_boxes = {}
    languages = {
        'Telugu': 'te',
        'Kannada': 'kn',
        'Malayalam': 'ml',
        'Hindi': 'hi'
    }

    # Create a frame for translation buttons
    btn_frame = Frame(trans_frame, bg=bg_color)
    btn_frame.pack(pady=5)

    # Create control buttons frame
    control_frame = Frame(left_frame, bg=bg_color)
    control_frame.pack(pady=5)

    # Speech frame for TTS buttons
    speech_frame = Frame(left_frame, bg=bg_color)
    speech_frame.pack(pady=5)

    # Create control buttons with consistent styling
    def create_button(parent, text, command, width=20):
        return Button(parent, text=text, command=command,
                      width=width,
                      bg=button_bg,
                      fg=button_fg,
                      activebackground='#3C3C3C',
                      activeforeground='white')

    # Create a frame for English TTS button
    english_speech_frame = Frame(speech_frame, bg=bg_color)
    english_speech_frame.pack(pady=2)

    # TTS button for English sentence
    tts_english_btn = create_button(english_speech_frame, "Speak English Sentence",
                                    lambda: text_to_speech(entryboxes['sentence_entrybox'].get('1.0', 'end-1c')))
    tts_english_btn.pack()

    # Create a frame for language TTS buttons
    language_speech_frame = Frame(speech_frame, bg=bg_color)
    language_speech_frame.pack(pady=2)

    # Function to create TTS callbacks
    def create_language_tts_callback(translation_box, lang_code):
        def tts_callback():
            text = translation_box.get('1.0', 'end-1c')
            text_to_speech(text, language=lang_code)

        return tts_callback

    # Function to create translation callbacks
    def create_translation_callback(sentence_box, trans_box, lang_code):
        def translate_callback():
            text = sentence_box.get('1.0', 'end-1c')
            if text:
                try:
                    translated = translator.translate(text, dest=lang_code)
                    trans_box.delete('1.0', 'end')
                    trans_box.insert('end', translated.text)
                except Exception as e:
                    trans_box.delete('1.0', 'end')
                    trans_box.insert('end', f"Translation error: {str(e)}")

        return translate_callback

    # TTS and translation buttons for each language
    for lang_name, lang_code in languages.items():
        # Create translation text box
        label_frame = Frame(trans_frame, bg=bg_color)
        label_frame.pack(pady=2)
        Label(label_frame, text=f"{lang_name}:", bg=bg_color, fg=fg_color).pack(side=LEFT)
        trans_box = Text(label_frame, height=2, width=20,
                         bg=entry_bg, fg=fg_color, insertbackground='white')
        trans_box.pack(side=LEFT, padx=5)
        translation_boxes[lang_name] = trans_box

        # Create translation button
        trans_btn = create_button(btn_frame, f"Translate to {lang_name}",
                                  create_translation_callback(entryboxes['sentence_entrybox'],
                                                              translation_boxes[lang_name],
                                                              lang_code))
        trans_btn.pack(pady=2)

        # Create TTS button for this language
        tts_btn = create_button(language_speech_frame, f"Speak {lang_name} Translation",
                                create_language_tts_callback(trans_box, lang_code))
        tts_btn.pack(side=LEFT, padx=2)

    # Add control buttons
    add_letter_btn = create_button(control_frame, "Add Letter",
                                   lambda: add_letter(current_predicted_char,
                                                      entryboxes['original word_entrybox'],
                                                      entryboxes['current char_entrybox']))
    add_letter_btn.pack(side=LEFT, padx=2)

    add_space_btn = create_button(control_frame, "Add Space",
                                  lambda: add_space(entryboxes['original word_entrybox'],
                                                    entryboxes['sentence_entrybox'],
                                                    entryboxes['corrected word_entrybox']))
    add_space_btn.pack(side=LEFT, padx=2)

    delete_btn = create_button(control_frame, "Delete",
                               lambda: delete_char(entryboxes['original word_entrybox']))
    delete_btn.pack(side=LEFT, padx=2)

    # Exit button
    exit_btn = create_button(left_frame, "Exit",
                             lambda: exit_app(gui, cap), width=15)
    exit_btn.pack(pady=5)

    # Start video processing
    names = ['vid_label', 'hands', 'th_box', 'cc_box', 'ow_box', 'cw_box', 'sent_box']
    with mp_hands.Hands(
            min_detection_confidence=0.4,
            min_tracking_confidence=0.5,
            max_num_hands=1) as hands:

        frame_video_stream(names, curr_char, prev_char, vid_label, hands,
                           entryboxes['threshold_entrybox'],
                           entryboxes['current char_entrybox'],
                           entryboxes['original word_entrybox'],
                           entryboxes['corrected word_entrybox'],
                           entryboxes['sentence_entrybox'])
        gui.root.mainloop()


# Main execution
title = "Sign Language Recognition GUI"
size = "1200x800"

gui, vid_label, left_frame = start_gui(title, size)
pipe_cam(gui, vid_label, left_frame)