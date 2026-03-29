from pydub import AudioSegment
import speech_recognition as sr
def fix_audio(input_file, output_file="fixed.wav"):
    print(input_file)
    audio = AudioSegment.from_file(input_file)
    audio = audio.set_channels(1)
    audio = audio.set_frame_rate(16000)
    audio.export(output_file, format="wav")

def Extract_Text(voicefilename):
 print(voicefilename)
 fix_audio(voicefilename)
 recognizer = sr.Recognizer()
 with sr.AudioFile("fixed.wav") as source:
    audio = recognizer.record(source)

 result = recognizer.recognize_google(audio, show_all=True)

 if result and 'alternative' in result:
    transcript = result['alternative'][0]['transcript']
 else:
    print("No speech detected")
 return transcript




