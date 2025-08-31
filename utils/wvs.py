import json


class WVS:
    def __init__(self, wave, language):
        self.wave = wave
        self.language = language
        self.data_list = self.read(self=self)

    @staticmethod
    def read(self):
        data_list = []
        file_path = f"dataset/questions/{self.wave}_{self.language}.jsonl"
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                data_list.append(json.loads(line))
        return data_list
    
