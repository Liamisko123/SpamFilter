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

if __name__ == "__main__":
    read_classification_from_file('/mnt/c/users/liams/uni/rph/programovanie/spam_filter/1/!truth.txt')