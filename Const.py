""" Constants for sequences """

# word pad and unk
PAD = 0
UNK = 1
PAD_WORD = '<pad>'
UNK_WORD = '<unk>'

# char pad and unk
cPAD = 0
cUNK = 1
PAD_CHAR = '<cpad>'
UNK_CHAR = '<cunk>'

# focused classes, emotion
four_emo = ['anger', 'joy', 'sadness', 'neutral']
four_iem = ['angry', 'happy', 'sad', 'neutral']

two_mosi = ['positive', 'negative']
rest_emo = ['surprise', 'disgust', 'fear', 'non-neutral']


# focused classes, review
five_yelp = ['1', '2', '3', '4', '5']
ten_imdb = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10']