import os
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter

dir = "C:\\Users\\USER\\Desktop\\my_pythonfiles_\\PythonForResearch\\book"
stats = pd.DataFrame(columns=("Language", "Author", "Title", "Length", "Unique"))
title_num = 1

def count_words(text):
    """
    counts the number of words in a given text and returns the unique words.
    Replace punctuation.
    """
    text = text.lower()
    split = [".", ",", ";", ":", "'", '"', "\n", "\r"]
    for ch in split:
        text = text.replace(ch, '')
    word_count = Counter(text.split(" "))
    return word_count

def read_book(book_path):
    """
    Reads a book from the given path, replaces special character
    """
    with open(book_path, 'r', encoding='utf8') as text:
        text = text.read()
        text = text.replace('\n', '').replace('\r', '')
    return text

def read_books(dir):
    """
    Read all books in the given path, replaces special character.
    """
    global title_num
    for language in os.listdir(dir):
        for author in os.listdir(dir + '\\' + language):
            if os.path.isdir(dir + '\\' + language + '\\' + author):
                for book in os.listdir(dir + '\\' + language + '\\' + author):
                    file_path = dir + '\\' + language + '\\' + author + '\\' + book
                    #print(file_path)
                    text = read_book(file_path)
                    (num_unique, counts) = word_stats(count_words(text))
                    stats.loc[title_num] = language, author.capitalize(), book.replace('.txt', ''), sum(counts), num_unique
                    title_num += 1

def word_stats(word_count):
    """Return number of unique words and word frequencies."""
    num_unique = len(word_count)
    counts = word_count.values()
    return (num_unique, counts)

def plot_stats():
    languages = ["English", "French", "German", "Portuguese"]
    color = ["crimson", "forestgreen", "orange", "blueviolet"]
    plt.figure(figsize=(10, 10))
    for i in range(4):
        subset = stats[stats.Language == languages[i]]
        plt.loglog(subset.Length, subset.Unique, "o", label=languages[i], color=color[i])
    plt.legend()
    plt.xlabel("Book length")
    plt.ylabel("Number of unique words")
    plt.savefig("Book Statistics.pdf")

def main():
    read_books(dir)
    print(stats["Length"])
    #plot_stats()
    #print(len(count_words("This comprehension check is to check for comprehension.")))
