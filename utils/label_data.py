# File Constants
START_LENGTH = 36

def read_file_from_line(file_path, start_line):
    try:
        with open(file_path, 'r') as file:
            # skipping the first start_line - 1 lines
            for _ in range(start_line - 1):
                next(file)
            # read the rest of the lines into a list
            lines = file.readlines()
        cleaned_list = [line.strip() for line in lines]
        return cleaned_list
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        return []
    
# Usage
POSITIVE_KEYWORDS = read_file_from_line('data/positive_words.txt', START_LENGTH)
NEGATIVE_KEYWORDS = read_file_from_line('data/negative_words.txt', START_LENGTH)
print(POSITIVE_KEYWORDS[:1])
print(NEGATIVE_KEYWORDS[:1])