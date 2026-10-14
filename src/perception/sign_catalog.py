"""Display names for the 30 released classes, from the project's QCVN 41:2024 copy.

Names do not establish vehicle/lane applicability or read numeric sign values.
For family-level labels, retain the code instead of inventing a subtype.
"""

NAMES_VI={
    'DP.134':'Hết tốc độ tối đa cho phép',
    'I.407a':'Đường một chiều',
    'I.409':'Chỗ quay xe',
    'P.102':'Cấm đi ngược chiều',
    'P.117':'Hạn chế chiều cao',
    'P.123a':'Cấm rẽ trái',
    'P.123b':'Cấm rẽ phải',
    'P.124a':'Cấm quay đầu xe',
    'P.124d':'Cấm rẽ phải và quay đầu xe',
    'P.125':'Cấm vượt',
    'P.127':'Tốc độ tối đa cho phép',
    'P.130':'Cấm dừng xe và đỗ xe',
    'P.131a':'Cấm đỗ xe',
    # Reviewed API support only; existing classifier checkpoints are unchanged.
    'P.131b':'Cấm đỗ xe ngày lẻ',
    'P.131c':'Cấm đỗ xe ngày chẵn',
    'P.137':'Cấm rẽ trái, rẽ phải',
    'R.122':'Dừng lại',
    'R.302a':'Hướng phải đi vòng chướng ngại vật',
    'R.420':'Bắt đầu khu đông dân cư',
    'R.421':'Hết khu đông dân cư',
    'R.303':'Nơi giao nhau chạy theo vòng xuyến',
    'W.201a':'Chỗ ngoặt nguy hiểm vòng bên trái',
    'W.203c':'Đường bị hẹp (biến thể c)',
    'W.207':'Giao nhau với đường không ưu tiên (nhóm)',
    'W.207b':'Giao nhau với đường không ưu tiên (biến thể b)',
    'W.208':'Giao nhau với đường ưu tiên',
    'W.210':'Giao nhau với đường sắt có rào chắn',
    'W.212':'Cầu hẹp',
    'W.220':'Dốc lên nguy hiểm',
    'W.222a':'Đường trơn',
    'W.224':'Đường người đi bộ cắt ngang',
    'W.225':'Trẻ em',
    'W.244':'Đoạn đường hay xảy ra tai nạn',
    'W.245a':'Đi chậm',
}

def display_name(code):
    if code=='unknown':return 'Chưa nhận dạng'
    name=NAMES_VI.get(code)
    return f'{code} — {name}' if name else code
