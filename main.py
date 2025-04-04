import requests
from bs4 import BeautifulSoup
import nltk
nltk.download("punkt")
from nltk.tokenize import sent_tokenize, word_tokenize
import spacy
import random
import csv

# 1. Собираем текст (~10 000 слов)
def collect_text():
    urls = [
        "https://ru.wikipedia.org/wiki/Москва",
        "https://ru.wikipedia.org/wiki/Санкт-Петербург",
        "https://ru.wikipedia.org/wiki/Яндекс",
        "https://ru.wikipedia.org/wiki/Сбербанк",
        "https://ru.wikipedia.org/wiki/Владимир_Путин",
        "https://ru.wikipedia.org/wiki/Российская_Федерация",
        "https://ru.wikipedia.org/wiki/Газпром"
    ]
    sentences = []
    for url in urls:
        response = requests.get(url)
        soup = BeautifulSoup(response.text, "html.parser")
        text = soup.find("div", class_="mw-parser-output").get_text()
        sents = sent_tokenize(text, language="russian")
        sentences.extend(sents[:150])  # Берем до 150 предложений с каждой страницы

    # Добавим несколько своих примеров для разнообразия
    custom_sentences = [
        "Петр Иванов работает в Яндексе в Москве.",
        "Анна Смирнова посетила Санкт-Петербург и Сбербанк.",
        "Владимир Путин встретился с представителями Газпрома в Кремле."
    ]
    sentences.extend(custom_sentences)
    return sentences

# 2. Предразметка с помощью spaCy (только PER, ORG, LOC)
nlp = spacy.load("ru_core_news_sm")

def annotate_sentence(sent):
    doc = nlp(sent)
    tokens = [token.text for token in doc]
    tags = ["O"] * len(tokens)
    
    # Разметка PER, ORG, LOC от spaCy
    for ent in doc.ents:
        start = ent.start
        end = ent.end
        tags[start] = f"B-{ent.label_}"
        for i in range(start + 1, end):
            tags[i] = f"I-{ent.label_}"
    
    # Фильтруем только слова (без символов и знаков препинания)
    filtered_tokens = [(token, tag) for token, tag in zip(tokens, tags) if token.isalpha()]
    
    return filtered_tokens

# 3. Создание CSV
sentences = collect_text()
random.shuffle(sentences)
sentences = sentences[:700]  # Ограничиваем до ~10 000 слов

data = []
word_count = 0
for sent in sentences:
    annotated = annotate_sentence(sent)
    data.extend(annotated)
    word_count += len(annotated)
    if word_count >= 10000:  # Останавливаемся на 10 000 слов
        break

# Сохранение в CSV
with open("ner_dataset.csv", "w", encoding="utf-8", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["token", "tag"])  # Заголовки
    writer.writerows(data)

print(f"Создан файл ner_dataset.csv с {word_count} словами.")
