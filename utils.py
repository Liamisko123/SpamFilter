import re
def read_classification_from_file(path):
    data = {}
    with open(path, 'r', encoding='UTF-8') as file:
        for line in file.readlines():
            name, validity = line.split()
            data[name] = validity
    return data

def write_classification_to_file(path, data):
    with open(path, 'w', encoding='UTF-8') as file:
        for key in data:
            file.write(f"{key} {data[key]}\n")


def contains_html(text):
    pattern = r'<[^>]*>'
    return bool(re.search(pattern, text))

def count_links(text):
    pattern = r'\b(?:https?|ftp):\/\/\S+'
    return len(re.findall(pattern, text))
