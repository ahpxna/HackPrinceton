"""Numeric explanation audit shared by rules and the optional model route."""
import re
from decimal import Decimal


def numeric_tokens(text):
    tokens=[Decimal(x.replace(',','')) for x in re.findall(r'(?<!\w)[+-]?(?:\d+(?:,\d{3})*(?:\.\d+)?|\.\d+)(?:[eE][+-]?\d+)?',text)]
    words={'zero':0,'one':1,'two':2,'three':3,'four':4,'five':5,'six':6,'seven':7,'eight':8,'nine':9,'ten':10,'eleven':11,'twelve':12,'thirteen':13,'fourteen':14,'fifteen':15,'sixteen':16,'seventeen':17,'eighteen':18,'nineteen':19,'twenty':20,'thirty':30,'forty':40,'fifty':50,'sixty':60,'seventy':70,'eighty':80,'ninety':90,'hundred':100,'thousand':1000,'million':1000000,'billion':1000000000}
    word=r'(?:'+'|'.join(words)+r')'
    for match in re.finditer(r'\b'+word+r'(?:(?:[\s-]+(?:and\s+)?)(?:'+word+r'))*\b',text,re.I):
        total=current=0
        for w in re.findall(word,match.group().lower()):
            value=words[w]
            if value==100:current=max(1,current)*value
            elif value>=1000:total+=max(1,current)*value;current=0
            else:current+=value
        tokens.append(Decimal(total+current))
    tokens += [Decimal(v) for w,v in [('half','0.5'),('quarter','0.25')] if re.search(r'\b'+w+r'\b',text,re.I)]
    return tokens


def audit_explanation(claim):
    operations={
        'round(value * 100, 1)':lambda v:round(v*100,1),
        'round(value / 60, 1)':lambda v:round(v/60,1),
        'round(value, 1)':lambda v:round(v,1),
        'round(value, 2)':lambda v:round(v,2),
        'round(value / 0.0254) // 12':lambda v:round(v/0.0254)//12,
        'round(value / 0.0254) % 12':lambda v:round(v/0.0254)%12,
        'abs(value)':abs,
    }
    for e in claim['evidence']:
        if not e['fact'].endswith('_display'):continue
        origin=e['derived_from'];operation=origin['operation']
        if operation not in operations or operations[operation](origin['value'])!=e['value']:
            raise ValueError('display number does not match its documented conversion')
        if not any(x['fact']==origin['fact'] and x['value']==origin['value'] and x['source_file']==e['source_file'] for x in claim['evidence']):
            raise ValueError('display number lacks its original numeric evidence')
    allowed={Decimal(str(e['value'])) for e in claim['evidence'] if isinstance(e.get('value'),(int,float)) and not isinstance(e['value'],bool)}
    if any(n not in allowed for n in numeric_tokens(claim['explanation'])):
        raise ValueError('explanation contains a number absent from tool evidence')
    if re.search(r'\b(?:convention|classifier|modelled|modeled)\b',claim['explanation'],re.I):
        raise ValueError('explanation contains buyer-facing jargon')
