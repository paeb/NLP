import re

with open('Texts/presentation.txt','r') as myfile: # the "with" operator -> the file is automatically closed at the end
    text = myfile.read()

print("1. Raw text")
print(text)
print()

# find the phone numbers

pattern = r'\d\d\d-\d\d\d-\d\d\d\d' # r denotes a special pattern recognition string
pattern = r'\d{3}-\d{3}-\d{4}' # using quantifiers to make the pattern searching more efficient

phone_numbers = re.findall(pattern,text) # . is a wildcard character (any char)

print("2. Phone numbers")
print(phone_numbers)
print()

# remove punctuation

new_text = re.findall(r"[^?.!:,]+",text)
new_text = ''.join(new_text) # joins together the items in sent with a space in between

print("3. Text without punctuation.")
print(new_text)