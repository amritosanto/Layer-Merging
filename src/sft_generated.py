"""Project-authored numeric curricula, independent of benchmark examples.

Templates cover arithmetic, geometry, motion and elementary probability.
Solutions are computed from recorded operands; no benchmark text is used.
"""
from fractions import Fraction
import random
from .utils import digest

VERSION='independent-numeric-curriculum-v1'
TEMPLATES={
 'English':[
  'There are {a} bags with {b} pencils each, plus {c} loose pencils. How many pencils are there?',
  'A shop has {a} boxes containing {b} pens each. It sells {c} pens. How many remain?',
  '{total} books are shared equally among {a} students. How many books does each student receive?',
  'One ticket costs {b} coins, with a discount of {c} coins per ticket. What is the cost of {a} tickets?',
  'A rectangle is {a} metres long and {b} metres wide. What is its perimeter in metres?',
  'A rectangle is {a} metres long and {b} metres wide. What is its area in square metres?',
  'A vehicle travels {total} kilometres in {a} hours at constant speed. What is its speed in kilometres per hour?',
  'A bag contains {a} red balls and {b} blue balls. What is the probability of drawing a red ball?'],
 'Hindi':[
  '{a} थैलों में प्रत्येक में {b} पेंसिल हैं और {c} पेंसिल अलग हैं। कुल कितनी पेंसिल हैं?',
  'एक दुकान में {a} डिब्बे हैं, प्रत्येक में {b} कलम हैं। {c} कलम बेचने के बाद कितनी बचेंगी?',
  '{total} पुस्तकें {a} विद्यार्थियों में बराबर बाँटी जाती हैं। प्रत्येक को कितनी पुस्तकें मिलेंगी?',
  'एक टिकट का मूल्य {b} सिक्के है और प्रति टिकट {c} सिक्के की छूट है। {a} टिकटों का मूल्य कितना होगा?',
  'एक आयत की लंबाई {a} मीटर और चौड़ाई {b} मीटर है। उसका परिमाप कितने मीटर है?',
  'एक आयत की लंबाई {a} मीटर और चौड़ाई {b} मीटर है। उसका क्षेत्रफल कितने वर्ग मीटर है?',
  'एक वाहन समान गति से {a} घंटे में {total} किलोमीटर चलता है। उसकी गति किलोमीटर प्रति घंटे में कितनी है?',
  'एक थैले में {a} लाल और {b} नीली गेंदें हैं। एक लाल गेंद निकालने की प्रायिकता क्या है?'],
 'Japanese':[
  '袋が{a}個あり、それぞれに鉛筆が{b}本入っています。ほかに鉛筆が{c}本あります。全部で何本ですか。',
  '店に箱が{a}個あり、それぞれにペンが{b}本入っています。{c}本売ると何本残りますか。',
  '{total}冊の本を{a}人の生徒に同じ冊数ずつ配ります。一人何冊ですか。',
  '切符は一枚{b}コインで、一枚につき{c}コインの割引があります。{a}枚の代金はいくらですか。',
  '長方形の縦は{a}メートル、横は{b}メートルです。周の長さは何メートルですか。',
  '長方形の縦は{a}メートル、横は{b}メートルです。面積は何平方メートルですか。',
  '車が一定の速さで{a}時間に{total}キロメートル進みます。時速は何キロメートルですか。',
  '袋に赤い球が{a}個、青い球が{b}個あります。赤い球を一つ取り出す確率はいくらですか。'],
 'Bengali':[
  '{a}টি ব্যাগের প্রতিটিতে {b}টি পেন্সিল আছে এবং আলাদা {c}টি পেন্সিল আছে। মোট কতটি পেন্সিল আছে?',
  'একটি দোকানে {a}টি বাক্সের প্রতিটিতে {b}টি কলম আছে। {c}টি কলম বিক্রির পরে কতটি থাকবে?',
  '{total}টি বই {a}জন শিক্ষার্থীর মধ্যে সমানভাবে ভাগ করা হলো। প্রত্যেকে কতটি বই পাবে?',
  'একটি টিকিটের দাম {b} মুদ্রা এবং প্রতিটি টিকিটে {c} মুদ্রা ছাড়। {a}টি টিকিটের দাম কত?',
  'একটি আয়তক্ষেত্রের দৈর্ঘ্য {a} মিটার এবং প্রস্থ {b} মিটার। পরিসীমা কত মিটার?',
  'একটি আয়তক্ষেত্রের দৈর্ঘ্য {a} মিটার এবং প্রস্থ {b} মিটার। ক্ষেত্রফল কত বর্গমিটার?',
  'একটি গাড়ি সমান গতিতে {a} ঘণ্টায় {total} কিলোমিটার চলে। ঘণ্টায় তার গতি কত কিলোমিটার?',
  'একটি ব্যাগে {a}টি লাল এবং {b}টি নীল বল আছে। একটি লাল বল তোলার সম্ভাবনা কত?'],
 'Swahili':[
  'Kuna mifuko {a}, kila mmoja una penseli {b}, pamoja na penseli {c} za ziada. Kuna penseli ngapi kwa jumla?',
  'Duka lina masanduku {a}, kila moja lina kalamu {b}. Likauza kalamu {c}. Zimebaki ngapi?',
  'Vitabu {total} vinagawanywa kwa usawa kati ya wanafunzi {a}. Kila mwanafunzi anapata vitabu vingapi?',
  'Tiketi moja inagharimu sarafu {b}, na punguzo la sarafu {c} kwa kila tiketi. Tiketi {a} zinagharimu kiasi gani?',
  'Mstatili una urefu wa mita {a} na upana wa mita {b}. Mzunguko wake ni mita ngapi?',
  'Mstatili una urefu wa mita {a} na upana wa mita {b}. Eneo lake ni mita za mraba ngapi?',
  'Gari linasafiri kilomita {total} kwa saa {a} kwa mwendo usiobadilika. Kasi yake ni kilomita ngapi kwa saa?',
  'Mfuko una mipira myekundu {a} na mipira ya bluu {b}. Uwezekano wa kuchagua mpira mwekundu ni upi?'],
 'Yoruba':[
  'Àpò {a} wà, ọ̀kọ̀ọ̀kan ní ohun èlò ìkọ̀wé {b}, àti {c} mìíràn. Mélòó ni ó wà lápapọ̀?',
  'Ilé ìtajà ní àpótí {a}, ọ̀kọ̀ọ̀kan ní kọ́bọ̀ {b}. Wọ́n mú kọ́bọ̀ {c} kúrò. Mélòó ló kù?',
  'A pín ìwé {total} fún akẹ́kọ̀ọ́ {a} ní ìwọ̀n kan náà. Mélòó ni ẹnìkọ̀ọ̀kan gba?',
  'Owó tíkẹ́ẹ̀tì kan jẹ́ {b}, a sì dín {c} kù lórí ọ̀kọ̀ọ̀kan. Mélòó ni owó tíkẹ́ẹ̀tì {a}?',
  'Onígun mẹ́rin gígùn kan ní gígùn mítà {a} àti fífẹ̀ mítà {b}. Mélòó ni àpapọ̀ gígùn gbogbo ẹ̀gbẹ́ rẹ̀?',
  'Onígun mẹ́rin gígùn kan ní gígùn mítà {a} àti fífẹ̀ mítà {b}. Mélòó ni ààyè rẹ̀ ní mítà onígun mẹ́rin?',
  'Ọkọ̀ kan rìn kìlómítà {total} ní wákàtí {a} pẹ̀lú eré kan náà. Mélòó ni eré rẹ̀ ní kìlómítà fún wákàtí kan?',
  'Àpò kan ní bọ́ọ̀lù pupa {a} àti bọ́ọ̀lù búlúù {b}. Kí ni ìṣeeṣe yíyan bọ́ọ̀lù pupa kan?'],
 'Telugu':[
  '{a} సంచులలో ఒక్కొక్కదానిలో {b} పెన్సిళ్లు ఉన్నాయి. అదనంగా {c} పెన్సిళ్లు ఉన్నాయి. మొత్తం ఎన్ని పెన్సిళ్లు ఉన్నాయి?',
  'ఒక దుకాణంలో {a} పెట్టెలలో ఒక్కొక్కదానిలో {b} పెన్నులు ఉన్నాయి. {c} పెన్నులు అమ్మితే ఎన్ని మిగులుతాయి?',
  '{total} పుస్తకాలను {a} మంది విద్యార్థులకు సమానంగా పంచారు. ఒక్కొక్కరికి ఎన్ని పుస్తకాలు వస్తాయి?',
  'ఒక టికెట్ ధర {b} నాణేలు, ఒక్కో టికెట్‌పై {c} నాణేల తగ్గింపు ఉంది. {a} టికెట్ల ధర ఎంత?',
  'ఒక దీర్ఘచతురస్రం పొడవు {a} మీటర్లు, వెడల్పు {b} మీటర్లు. దాని చుట్టుకొలత ఎన్ని మీటర్లు?',
  'ఒక దీర్ఘచతురస్రం పొడవు {a} మీటర్లు, వెడల్పు {b} మీటర్లు. దాని వైశాల్యం ఎన్ని చదరపు మీటర్లు?',
  'ఒక వాహనం స్థిర వేగంతో {a} గంటలలో {total} కిలోమీటర్లు ప్రయాణిస్తుంది. గంటకు దాని వేగం ఎన్ని కిలోమీటర్లు?',
  'ఒక సంచిలో {a} ఎరుపు బంతులు, {b} నీలం బంతులు ఉన్నాయి. ఒక ఎరుపు బంతిని తీసే సంభావ్యత ఎంత?']}


def solve(kind,a,b,c):
    return [a*b+c,a*b-c,b,a*(b-c),2*(a+b),a*b,b,Fraction(a,a+b)][kind]


def generate(language,category,index,seed,label_index=0):
    rng=random.Random(int(digest([VERSION,seed,language,category,index])[:16],16))
    a=rng.randint(2,70);b=rng.randint(6,400);c=rng.randint(1,b-1)
    kind=index%(7 if category=='math' else 8)
    answer=solve(kind,a,b,c);total=a*b
    question=TEMPLATES[language][kind].format(a=a,b=b,c=c,total=total)
    expressions=[f'{a} × {b} + {c}',f'{a} × {b} - {c}',f'{total} ÷ {a}',f'{a} × ({b} - {c})',f'2 × ({a} + {b})',f'{a} × {b}',f'{total} ÷ {a}',f'{a} / ({a} + {b})']
    response=f'{expressions[kind]} = {answer}.\n#### {answer}'
    choices=None;label=None
    if category=='mc':
        # Rotate the accepted-record label, not the attempted-record index.
        label=label_index%4
        wrong=[answer+Fraction(i,1 if kind!=7 else (a+b)*2) for i in (1,2,3)]
        choices=[str(x) for x in wrong];choices.insert(label,str(answer))
        question='Question:\n'+question+'\n\n'+'\n'.join(f'{chr(65+i)}. {v}' for i,v in enumerate(choices))+'\n\nAnswer:'
        response=' '+chr(65+label)
        assert len(set(choices))==4 and choices[label]==str(solve(kind,a,b,c))
    return dict(inputs=question,targets=response,language=language,category=category,
        source=VERSION,annotation_type='project_verified',problem_id=digest([kind,a,b,c if kind in (0,1,3) else None]),
        verified_answer=str(answer),operands=[kind,a,b,c],choices=choices,label=label,
        subject=['arithmetic','arithmetic','arithmetic','arithmetic','geometry','geometry','physics','probability'][kind])
