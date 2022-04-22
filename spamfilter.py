#!/usr/bin/env python
from io import TextIOWrapper
from math import log


class Szo:
    """Cucc"""

    spam_feltunes: float = 0.0
    ham_feltunes: float = 0.0

    def __str__(self) -> str:
        return "Spam: {} Ham: {}".format(self.spam_feltunes, self.ham_feltunes)


TRAIN_INPUT: str = "input/train.txt"
TEST_INPUT: str = "input/test.txt"
STOP_WORDS_INPUT: str = "input/stopwords2.txt"

stop_words: list[str] = []
spam_files_train: list[str] = []
ham_files_train: list[str] = []

szo_dict: dict[str, Szo] = dict()

spam_word_count: int = 0
ham_word_count: int = 0

ALFA: float = 0.01


def get_filtered_text(file_input: TextIOWrapper) -> list[str]:
    text = list(
        filter(
            lambda x: x not in stop_words,
            "".join(file_input.read().splitlines()).split(" "),
        )
    )
    return text


def is_spam(spam_prob: float, ham_prob: float, document: str) -> bool:
    card_dict: dict[str, int] = dict()
    with open(document, "r", errors="ignore") as doc_input:
        text = get_filtered_text(doc_input)
    for word in text:
        if word in card_dict:
            card_dict[word] += 1
        else:
            card_dict[word] = 1
    logR = log(spam_prob) - log(ham_prob)
    for key in card_dict:
        if key in szo_dict:
            logR += card_dict[key] * (
                log(szo_dict[key].spam_feltunes) - log(szo_dict[key].ham_feltunes)
            )
    return True if logR > 0 else False


with open(STOP_WORDS_INPUT, "r") as stop_input:
    stop_words = stop_input.read().splitlines()

stop_words = stop_words + ["Subject:", ";", ",", ".", "!", "'", "-", ":", "/", "&"]

with open(TRAIN_INPUT, "r") as test_input:
    for test_file in test_input.read().splitlines():
        if "ham" in test_file:
            ham_files_train.append("input/ham/{}".format(test_file))
        else:
            spam_files_train.append("input/spam/{}".format(test_file))

for ham_file in ham_files_train:
    with open(ham_file, "r", encoding="utf-8", errors="ignore") as ham_input:
        text = get_filtered_text(ham_input)
    for word in text:
        if word in szo_dict:
            szo_dict[word].ham_feltunes = szo_dict[word].ham_feltunes + 1.0
        else:
            szo_dict[word] = Szo()
            szo_dict[word].ham_feltunes = 1.0
    ham_word_count += len(text)

for spam_file in spam_files_train:
    with open(spam_file, "r", encoding="utf-8", errors="ignore") as spam_input:
        text = get_filtered_text(spam_input)
    for word in text:
        if word in szo_dict:
            szo_dict[word].spam_feltunes = szo_dict[word].spam_feltunes + 1.0
        else:
            szo_dict[word] = Szo()
            szo_dict[word].spam_feltunes = 1.0
    spam_word_count += len(text)

KEY_SIZE: int = len(szo_dict.keys())

for key in szo_dict:
    spam_card = szo_dict[key].spam_feltunes
    szo_dict[key].spam_feltunes = (spam_card + ALFA) / (
        ALFA * KEY_SIZE + spam_word_count
    )
    ham_card = szo_dict[key].ham_feltunes
    szo_dict[key].ham_feltunes = (ham_card + ALFA) / (ALFA * KEY_SIZE + ham_word_count)

spam_label_count = 0
ham_label_count = 0
spam_prob = len(spam_files_train) / (len(spam_files_train) + len(ham_files_train))
ham_prob = len(ham_files_train) / (len(spam_files_train) + len(ham_files_train))

for spam_file in spam_files_train:
    if is_spam(spam_prob, ham_prob, spam_file):
        spam_label_count += 1

for ham_file in ham_files_train:
    if not is_spam(spam_prob, ham_prob, ham_file):
        ham_label_count += 1


print("Tanulasi hibak:")
print("Spam hiba: {}%".format(100 - (spam_label_count * 100) / len(spam_files_train)))
print("Ham hiba: {}%".format(100 - (ham_label_count * 100) / len(ham_files_train)))

spam_files_test: list[str] = []
ham_files_test: list[str] = []
spam_label_count = 0
ham_label_count = 0

with open(TEST_INPUT, "r") as test_input:
    for test_file in test_input.read().splitlines():
        if "ham" in test_file:
            ham_files_test.append("input/ham/{}".format(test_file))
        else:
            spam_files_test.append("input/spam/{}".format(test_file))

for spam_file in spam_files_test:
    if is_spam(spam_prob, ham_prob, spam_file):
        spam_label_count += 1

for ham_file in ham_files_test:
    if not is_spam(spam_prob, ham_prob, ham_file):
        ham_label_count += 1


print("Tesztelesi hibak:")
print("Spam hiba: {}%".format(100 - (spam_label_count * 100) / len(spam_files_test)))
print("Ham hiba: {}%".format(100 - (ham_label_count * 100) / len(ham_files_test)))
