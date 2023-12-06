from corpus import Corpus

def test_reading_folder():
    folder_path = '/mnt/c/users/liams/uni/RPH/programovanie/spam_filter/1'
    corp = Corpus(folder_path)
    for file_name, content in corp.emails():
        print(f"{file_name}:  {content[:20]}")
        
test_reading_folder()