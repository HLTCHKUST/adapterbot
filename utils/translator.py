from googletrans import Translator
import googletrans

translator = Translator()
def translate(sentence,src,trg,auto=False):
    if (src != 'en' and trg == 'en') or (src == 'en' and trg != 'en'):
        print(src)
        print(trg)
        sentence = translator.translate(sentence, src=src, dest=trg).text.lower()
        print(f"trainlated:{sentence}")
    return sentence

