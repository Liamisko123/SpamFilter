import os

class Corpus:

    def __init__(self, path) -> None:
        self.path = path

    def emails(self):
        for file_name in os.listdir(self.path):
            if file_name[0] == '!':
                continue
            with open(os.path.join(self.path, file_name), 'r', encoding='UTF-8') as file:
                content = file.read()
            yield file_name, content

    def __str__(self) -> str:
        return "Lebron James"