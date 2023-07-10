# Different tokens lengths
# sentence_pairs = [
#     ("I am eating an apple.", "Je mange une pomme."),
#     ("She reads the book.", "Elle lit le livre."),
#     ("He is walking in the park.", "Il se promène dans le parc."),
#     ("We are studying French.", "Nous étudions le français."),
#     ("They are watching a movie.", "Ils regardent un film."),
#     ("You are drinking water.", "Tu bois de l'eau."),
#     ("The cat is sleeping on the chair.", "Le chat dort sur la chaise."),
#     ("The children are playing in the garden.", "Les enfants jouent dans le jardin."),
#     ("He loves her.", "Il l'aime."),
#     ("She is going to school.", "Elle va à l'école.")
# ]


# API that doesn't work
# from googletrans import Translator
# translator = Translator()
# english_data = haystack_utils.load_txt_data("data/kde4_english.txt")

# print(translator.translate("hello", src='en', dest='fr'))

# same_token_len_pairs_large = []
# for item in english_data:
#     for word in item.split(" "):
#         try:
#             translation = translator.translate(word, src='en', dest='fr')
#         except:
#             continue
#         if translation is None or translation.text is None:
#             continue

#         if model.to_tokens(" " + word).shape[1] == model.to_tokens(" " + translation.text).shape[1]:
#             same_token_len_pairs_large.append((word, translation.text))
#         if len(same_token_len_pairs_large) == 100:
#             break