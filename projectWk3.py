# CASE STUDY 2

import pandas as pd
import matplotlib.pyplot as plt
from languageProcessing import count_words

ind = 1
frequency = ""
hamlets = pd.read_csv(r"C:\Users\USER\Desktop\my_pythonfiles_\PythonForResearch\book\hamlets.csv", index_col=0)
language, text = hamlets.iloc[0]
counted_text = count_words(text)
sub_data = pd.DataFrame(columns=("language", "frequency", "mean_word_length", "num_words"))
data = pd.DataFrame(columns=("word", "count", "length", "frequency"))
def main1():
    global ind
    for (word, count) in counted_text.items():
        length = len(word)
        if count == 1:
            frequency = "unique"
        elif count > 10:
            frequency = "frequent"
        elif count in range(2,11):
            frequency = "infrequent"
        data.loc[ind] = word, count, length, frequency
        ind += 1

def main2():
    ind = 1
    frequency = ["frequent", "infrequent", "unique"]
    for often in frequency:
        mean = data[data.frequency == often].loc[:,"length"].mean()
        num_words = len(data[data.frequency == often])
        sub_data.loc[ind] = language, often, mean, num_words
        ind += 1
    return (sub_data)
        

def summarize_text(language, text):
    counted_text = count_words(text)

    data = pd.DataFrame({
            "word":list(counted_text.keys()),
            "count":list(counted_text.values())
        })

    data.loc[data["count"] > 10, "frequency"] = "frequent"
    data.loc[data["count"] <= 10, "frequency"] = "infrequent"
    data.loc[data["count"] == 1, "frequency"] = "unique"

    data["length"] = data["word"].apply(len)

    sub_data = pd.DataFrame({
        "language":language,
        "frequency":["frequent", "infrequent","unique"],
        "mean_word_length":data.groupby(by = "frequency")["length"].mean(),
        "num_words":data.groupby(by="frequency").size()
    }, )

    return (sub_data)


all_df = []
#grouped_data = pd.DataFrame(columns=("languages", "frequency", "mean_word_length", "num_words"))
for i in range(3):
    language, text = hamlets.iloc[i]
    sub_data = summarize_text(language, text)
    all_df.append(sub_data)
    #print(sub_data.head())
    #grouped_data = grouped_data.append(sub_data)
grouped_data = pd.concat(all_df)

print(grouped_data)

def plot():
    colors = {"Portuguese":"green", "English":"blue", "German":"red"}
    markers = {"frequent":"o", "infrequent":"s", "unique":"^"}
    for i in range(grouped_data.shape[0]):
        row = grouped_data.iloc[i]
        plt.plot(row.mean_word_length, row.num_words, 
                 marker=markers[row.frequency], color = colors[row.language
                ])

    color_legend = []
    marker_legend = []
    for color in colors:
        color_legend.append(plt.plot([],[], color=colors[color], marker="o",
                label=color, markersize=10, linestyle="None")
            )
    for marker in markers:
        marker_legend.append(plt.plot([], [], color="k", marker=markers[marker],
                label=marker, markersize=10, linestyle="None")
            )
    plt.legend(numpoints=1, loc="upper left")
    plt.xlabel("Mean Word Length")
    plt.ylabel("Number of Words")
    plt.savefig("languageStatistics.pdf")

plot()




#num_words = len(data[data.frequency == "unique"])