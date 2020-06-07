import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from projectWk2 import play_strategic_game

def main():
    sns.set()
    results = []
    for _ in range(1000):
        results.append(play_strategic_game())
    result_pie = [results.count(i) for i in range(-1, 3)]
    print(result_pie)
    plt.pie(result_pie, labels=[-1, 0, 1, 2])
    plt.show()
    #plt.savefig("tictactoe.pdf")

main()