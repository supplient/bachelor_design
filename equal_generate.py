from preprocess import load_file
import translate

class EqualGenerator:
    def __init__(self, src="zh-cn", mid="en"):
        self._to_translator = translate.Translator("en", from_lang="zh-cn")
        self._back_translator = translate.Translator("zh-cn", from_lang="en")
        self._src = src
        self._mid = mid

    def generate(self, text_list):
        from tqdm import tqdm
        for text in tqdm(text_list, desc="Generating"):
            mid_text = self._to_translator.translate(text)
            equal_text = self._back_translator.translate(mid_text)
            yield equal_text


if __name__ == "__main__":
    generator = EqualGenerator()
    a = ["你好", "晚安", "请输入一个整数"]
    b = [t for t in generator.generate(a)]
    for c, d in zip(a, b):
        print(c)
        print(d)
        print()