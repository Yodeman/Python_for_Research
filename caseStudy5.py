import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

input_file = "bird_tracking.csv"
#output_file = open("bird.txt", 'w')
data = pd.read_csv(input_file)
#print(data.head())
#print(data["date_time"][5])
grouped_birds = data.groupby("bird_name")
#print(grouped_birds.head())
mean_speeds = grouped_birds["speed_2d"].mean()
#print(mean_speeds)
mean_altitudes = grouped_birds["altitude"].mean()
#print(mean_altitudes)
data.date_time = pd.to_datetime(data["date_time"])
#print(data["date_time"][5])
data["date"] = data["date_time"].dt.date
#print(data["date"][5])
grouped_bydates = data.groupby("date")
mean_altitudes_perday = grouped_bydates["altitude"].mean()
date = pd.to_datetime("2014-04-04").date()
#print(date)
#print(mean_altitudes_perday[date])
grouped_birdday = data.groupby(["bird_name", "date"])
#print(grouped_birdday.head())
mean_altitude_perday = grouped_birdday["altitude"].mean()
#print(mean_altitude_perday.tail(n=20))
mean_speed_perday = grouped_birdday["speed_2d"].mean()
eric_daily_speed = mean_speed_perday["Eric"]
#print(eric_daily_speed[:5])
sanne_daily_speed = mean_speed_perday["Sanne"]
nico_daily_speed = mean_speed_perday["Nico"]
print(nico_daily_speed.tail(n=30))
eric_daily_speed.plot(label="Eric")
sanne_daily_speed.plot(label="Sanne")
nico_daily_speed.plot(label="Nico")
plt.legend(loc="upper left")
plt.show()
