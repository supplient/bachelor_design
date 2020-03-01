_locale = False
try:
    import google.colab
    google.colab.drive.mount("/gdrive")
except ModuleNotFoundError:
    _locale = True

addh = "/gdrive/My Drive"
if _locale:
    addh = "/mnt/d/My Drive"
    print("[Locale] Using address head: " + addh)
else:
    print("[Colab] Using address head: " + addh)