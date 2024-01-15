"""
view https://github.com/adbar/trafilatura for detail
"""

from goose3 import Goose
from goose3.text import StopWordsChinese

if __name__ == '__main__':
    url  = 'https://www.zgbk.com/ecph/words?SiteID=1&ID=220442&Type=bkzyb&SubID=147627'
    g = Goose({'stopwords_class': StopWordsChinese})
    article = g.extract(url=url)
    print(article.cleaned_text)

    print(article.authors, article.doc, article.tags)


