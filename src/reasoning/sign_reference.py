from pathlib import Path
import json

class SignReference:
    def __init__(self,path='data/processed/legal_reference/index.json'):
        p=Path(path)
        self.entries=json.loads(p.read_text(encoding='utf-8'))['entries'] if p.is_file() else {}

    def lookup(self,codes):
        return [self.entries[code] for code in dict.fromkeys(codes) if code in self.entries]
