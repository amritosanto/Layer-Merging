import re
import unicodedata
from decimal import Decimal, InvalidOperation
from fractions import Fraction

LANGUAGES=['en','es','fr','de','ru','zh','ja','th','sw','bn','te']
PREFIXES={'en':('Question','Answer'),'es':('Pregunta','Respuesta'),'fr':('Question','Réponse'),
          'de':('Frage','Antwort'),'ru':('Вопрос','Ответ'),'zh':('问题','答案'),'ja':('問題','答え'),
          'th':('คำถาม','คำตอบ'),'sw':('Swali','Jibu'),'bn':('প্রশ্ন','উত্তর'),'te':('ప్రశ్న','సమాధానం')}

def normalized_number(text,language='en'):
    text=unicodedata.normalize('NFKC',str(text)).replace('−','-').replace('٫','.').replace('٬',',')
    text=''.join(str(unicodedata.digit(c)) if c.isdigit() else c for c in text)
    text=re.sub(r'(?<=\d)[ \u00a0\u202f](?=\d)','',text).strip()
    if ',' in text and '.' in text:
        if text.rfind(',')>text.rfind('.'):text=text.replace('.','').replace(',','.')
        else:text=text.replace(',','')
    elif ',' in text:
        if re.fullmatch(r'[-+]?\d{1,3}(,\d{3})+',text):text=text.replace(',','')
        else:text=text.replace(',','.')
    elif language in ('de','es') and re.fullmatch(r'[-+]?\d{1,3}(\.\d{3})+',text):
        text=text.replace('.','')
    try:
        value=Fraction(text) if '/' in text else Fraction(Decimal(text))
        return str(value)
    except (ValueError,InvalidOperation,ZeroDivisionError):return None

def extract_answer(text,language='en',complete=True):
    # Stop at any subsequent question, including translated question prefixes.
    questions='|'.join(re.escape(q) for q,_ in PREFIXES.values())
    boundary=re.search(r'(?im)^\s*(?:'+questions+r')\s*[:：]',text)
    if boundary:text=text[:boundary.start()]
    matches=list(re.finditer(r'####[ \t]*([^\r\n]*)',text))
    for match in matches:
        if match.end()==len(text) and not complete and boundary is None:continue
        value=match.group(1).strip()
        if re.fullmatch(r'[-+−]?\d+(?:[.,٫٬\u00a0\u202f]\d+)*(?:/\d+)?[.!]?',value):
            answer=normalized_number(value.rstrip('.!'),language)
            if answer is not None:return answer
    if matches or not complete or boundary:return None
    lines=[line.strip() for line in text.splitlines() if line.strip()]
    if not lines:return None
    tail=lines[-1]
    if not re.fullmatch(r'[-+−]?\d+(?:[.,٫٬\u00a0\u202f]\d+)*(?:/\d+)?[.!]?',tail):return None
    return normalized_number(tail.rstrip('.!'),language)


def prompt(support,question,language):
    q,a=PREFIXES[language]
    examples=[f'{q}: {r["question"]}\n{a}: {r["answer"]}\n#### {r["answer_number"]}' for r in support]
    return '\n\n'.join(examples+[f'{q}: {question}\n{a}:'])
