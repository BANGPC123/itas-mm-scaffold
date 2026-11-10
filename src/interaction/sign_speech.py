"""Short, non-duplicated Vietnamese TTS for reviewed and applicable signs only.

Do not turn an uncertain prediction into an audible driving instruction.
"""
from src.perception.sign_catalog import NAMES_VI


def sign_speech(signs):
    messages = set()
    vehicles = {'car': 'ô tô', 'truck': 'xe tải', 'motorcycle': 'xe máy',
                'motorbike':'xe máy','bus':'xe buýt','van':'xe van','bicycle':'xe đạp'}
    for sign in signs:
        if not (sign.applicability == 'applies' and sign.context_verified
                and sign.label_verified and sign.panel_role == 'primary'
                and sign.label in NAMES_VI):
            continue
        if sign.label in {'P.127', 'P.117'}:
            # Family codes lack the numeric speed/height value in our
            # current recognizer: an incomplete audible instruction is unsafe.
            continue
        if sign.is_compound and (not sign.panel_labels or (
                not sign.allowed_vehicles and not sign.time_conditions and any(s not in NAMES_VI for s in sign.panel_labels[1:]))):
            # A supplementary plate has not been decoded as a supported
            # QCVN class/condition; the main sign is not a complete command.
            continue
        name = NAMES_VI[sign.label]
        if sign.allowed_vehicles:
            if any(v not in vehicles for v in sign.allowed_vehicles):
                continue
            name += '; dành cho ' + ', '.join(vehicles[v] for v in sorted(sign.allowed_vehicles))
        if sign.time_conditions:
            name += '; trong khung giờ ' + ' và '.join(' hoặc '.join(a+' đến '+b for a,b in windows) for windows in sign.time_conditions)
        messages.add('Biển áp dụng cho làn đang đi: ' + name + '.')
    return ' '.join(sorted(messages))
