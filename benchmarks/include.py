LANGUAGE_CODES=dict(zip(
    'Albanian Arabic Armenian Azerbaijani Basque Belarusian Bengali Bulgarian Chinese Croatian Dutch Estonian Finnish French Georgian German Greek Hebrew Hindi Hungarian Indonesian Italian Japanese Kazakh Korean Lithuanian Macedonian Malay Malayalam Nepali Persian Polish Portuguese Russian Serbian Spanish Tagalog Tamil Telugu Turkish Ukrainian Urdu Uzbek Vietnamese'.split(),
    'sq ar hy az eu be bn bg zh hr nl et fi fr ka de el he hi hu id it ja kk ko lt mk ms ml ne fa pl pt ru sr es tl ta te tr uk ur uz vi'.split()))
LANGUAGE_CODES.update({'Dutch-Flemish':'nl','North Macedonian':'mk'})

def adapt(row,index):
    lowered={k.lower():v for k,v in row.items()}
    question=lowered.get('question'); choices=lowered.get('choices',lowered.get('options'))
    answer=lowered.get('answer',lowered.get('answer_index'))
    if isinstance(choices,dict):choices=[choices[k] for k in sorted(choices)]
    if not isinstance(question,str) or not isinstance(choices,(list,tuple)) or len(choices)<2:
        raise ValueError(f'Unsupported INCLUDE schema: {list(row)}')
    if isinstance(answer,str) and answer.strip() in list('ABCDEFGHIJKLMNOPQRSTUVWXYZ'):
        answer=ord(answer.strip())-ord('A')
    if not isinstance(answer,int) or not 0<=answer<len(choices):raise ValueError(f'Invalid INCLUDE answer: {answer}')
    return dict(id=str(index),question=question,choices=list(choices),label=answer,
                subject=lowered.get('subject','unknown'),domain=lowered.get('domain','unknown'),
                regional=lowered.get('regional_feature','unknown'))
