# Installing nltk

To install nltk, you will need to first import the nltk library and run the nltk.download() file. 

```
import nltk
nltk.download()
```

If you are having issues installing nltk and it’s due to SSL certifications, run the following code on the terminal or an IDE. It is recommend to save the code as a file and run it via the command line or an IDE. Source code: https://stackoverflow.com/questions/38916452/nltk-download-ssl-certificate-verify-failed

```
import nltk
import ssl

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

nltk.download()
```