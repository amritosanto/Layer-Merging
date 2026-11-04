LOCALES=['AR_XY','BN_BD','DE_DE','ES_LA','FR_FR','HI_IN','ID_ID','IT_IT','JA_JP','KO_KR','PT_BR','SW_KE','YO_NG','ZH_CN']

def adapt(row,index):
    choices=[row[k] for k in 'ABCD']; answer=str(row['Answer']).strip()
    if answer not in 'ABCD' or len(answer)!=1:raise ValueError(f'Invalid MMMLU label {answer}')
    return dict(id=str(index),question=row['Question'],choices=choices,label='ABCD'.index(answer),
                subject=row.get('Subject','unknown'),domain=row.get('Subject','unknown'),regional='unknown')
