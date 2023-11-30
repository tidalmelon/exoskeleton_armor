import re


def remove_blank_keep_newline(text, pat=re.compile(r"['\xa0', u'\u3000', ' ', u'\u202f', '\t', '\f', '\v']+")):
    return pat.sub(' ', text)

