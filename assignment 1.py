def my_split(text, sep):
    parts = []
    word = ""
    for ch in text:
        if ch != sep:
            word += ch
        else:
            parts.append(word)
            word = ""
    if word:
        parts.append(word)
    return parts

def my_join(lst, sep):
    result = ""
    for i in range(len(lst)):
        result += lst[i]
        if i < len(lst) - 1:
            result += sep
    return result

line = input("Please enter the sentence: ")
words = my_split(line, " ")

print(my_join(words, ","))
for w in words:
    print(w)
