import config

_locale = False
try:
    import google.colab
    google.colab.drive.mount(config.COLAB_AMOUNT_PATH)
except ModuleNotFoundError:
    _locale = True

addh = config.COLAB_ADDH
if _locale:
    addh = config.LOCAL_ADDH
    print("[Locale] Using address head: " + addh)
else:
    print("[Colab] Using address head: " + addh)