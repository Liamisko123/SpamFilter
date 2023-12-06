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