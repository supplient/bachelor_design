class Translator:
    def translate(self, text, src, dest):
        raise NotImplementedError

class SougoTranslator(Translator):
    def _build_payload(self, src, dest, query):
        import hashlib
        import  urllib.parse
        import time
        import secret

        def md5(str):
            m = hashlib.md5()
            m.update(str.encode("utf8"))
            return m.hexdigest()

        pid = secret.SougoTrans.app_id
        key = secret.SougoTrans.key
        salt = str(time.time())     #随机数，可以填入时间戳
        q = query
        sign = md5(pid+q+salt+key)  

        #可查看语种列表替换语种代码
        src = src
        to = dest
        payload = "from=" + src + "&to=" + to + "&pid=" + pid + "&q=" +urllib.parse.quote(q) + "&sign=" + sign + "&salt=" + salt
        return payload

    def translate(self, text, src, dest):
        import requests
        import json

        if src == "zh":
            src = "zh-CHS"
        if dest == "zh":
            dest = "zh-CHS"

        result = ""

        url = "http://fanyi.sogou.com:80/reventondc/api/sogouTranslate"     #请求的接口地址
        payload = self._build_payload(src, dest, text)
        headers = {
            'content-type': "application/x-www-form-urlencoded",
            'accept': "application/json"
            }
        try:
            response = requests.request("POST", url, data=payload, headers=headers)
            response_json = json.loads(response.text)
            result = response_json["translation"]
        except Exception as e:
            import logging
            logging.warn(e)
        return result


class BaiduTranslator(Translator):

    def _build_url(self, query, src, dest):
        import hashlib
        import urllib
        import random
        import secret
        appid = secret.BaiduTrans.app_id
        secretKey = secret.BaiduTrans.key

        myurl = '/api/trans/vip/translate'

        fromLang = src
        toLang = dest
        salt = random.randint(32768, 65536)
        q= query
        sign = appid + q + str(salt) + secretKey
        sign = hashlib.md5(sign.encode()).hexdigest()
        myurl = myurl + '?appid=' + appid + '&q=' + urllib.parse.quote(q) + '&from=' + fromLang + '&to=' + toLang + '&salt=' + str(
                salt) + '&sign=' + sign
        return myurl

    def translate(self, text, src, dest):
        import http.client
        import json

        httpClient = None
        url = self._build_url(text, src, dest)
        result = ""

        try:
            httpClient = http.client.HTTPConnection('api.fanyi.baidu.com')
            httpClient.request('GET', url)

            # response是HTTPResponse对象
            response = httpClient.getresponse()
            result_all = response.read().decode("utf-8")
            result = json.loads(result_all)["trans_result"][0]["dst"]
        except Exception as e:
            import logging
            logging.warning(e)
        finally:
            if httpClient:
                httpClient.close()
        
        return result

class EqualGenerator:
    def __init__(self, src="zh", mid="en"):
        self._src = src
        self._mid = mid
        self._translator = SougoTranslator()

    def generate(self, text):
        mid_text = self._translator.translate(text, self._src, self._mid)
        equal_text = self._translator.translate(mid_text, self._mid, self._src)
        return equal_text

    @classmethod
    def test(cls):
        import tqdm
        generator = EqualGenerator()
        a = ["你好", "晚安", "请输入一个整数"]
        b = [generator.generate(t) for t in tqdm.tqdm(a)]
        for c, d in zip(a, b):
            print(c)
            print(d)
            print()

def work(max_count):
    import logging

    from preprocess import load_file, seq2str, str2seq
    from driver_amount import addh
    import config
    logging.info("Loading origin data...")
    char_seqs, tag_seqs = load_file(addh + config.DATA_PATH)
    equal_generator = EqualGenerator()

    def save(filepath, obj, count):
        import json
        import os
        dirname = os.path.dirname(filepath)
        count_path = os.path.join(dirname, "count.txt")
        with open(filepath, "w") as fd:
            json.dump(obj, fd)
        with open(count_path, "w") as fd:
            fd.write(str(count))

    from tqdm import tqdm
    logging.info("Start generating equal data.")
    equal_seqs = []
    start = 0
    count = 0
    for char_seq in tqdm(char_seqs, total=max_count):
        if count < start:
            count += 1
            continue
        origin_str = seq2str(char_seq)
        equal_str = equal_generator.generate(origin_str)
        equal_seq = str2seq(equal_str)
        equal_seqs.append(equal_seq)

        count += 1
        if count % 50 == 0:
            logging.info("Save " + str(int(count/1000)) + "th")
            save(
                addh + config.EQUAL_DATA_PATH,
                equal_seqs,
                count
            )
            logging.info("Save Done.")

        if count >= max_count:
            break

    logging.info("Save the rest")
    save(
        addh + config.EQUAL_DATA_PATH,
        equal_seqs,
        count
    )

    

if __name__ == "__main__":
    # EqualGenerator.test()
    work(100)