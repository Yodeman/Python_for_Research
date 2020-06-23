import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

bird_data = pd.read_csv("bird_tracking.csv")

# Bird Trajectory
"""bird_names = pd.unique(bird_data.bird_name)
plt.figure()
for bird_name in bird_names:
    ix = bird_data.bird_name == bird_name
    x, y = bird_data.longitude[ix], bird_data.latitude[ix]

    plt.plot(x, y, ".", label=bird_name)
plt.xlabel("longitude")
plt.ylabel("latitude")
plt.legend(loc="lower right")
plt.savefig("birdTraj.pdf")
plt.show()"""

# Flight speed
"""ix = bird_data.bird_name=="Eric"
speed = bird_data.speed_2d[ix]
#print(speed[:5])
ind = np.isnan(speed)
plt.hist(speed[~ind], bins=np.linspace(0, 30, 20), density=True)
plt.xlabel("2D speed (m/s)")
plt.ylabel("Frequency")
plt.show()"""

# using pandas to plot the data.
"""bird_data.speed_2d.plot(kind="hist", range=[0, 30])
plt.xlabel("2D speed")
plt.show()
"""

# Plotting data_time
import datetime as dt

time_stamps = []
for k in range(len(bird_data)):
    time_stamps.append(dt.datetime.strptime(bird_data.date_time.iloc[k][:-3], "%Y-%m-%d %H:%M:%S"))

bird_data["time_stamp"] = pd.Series(time_stamps, index=bird_data.index)
#print(bird_data.head())
times = bird_data.time_stamp[bird_data.bird_name == "Sanne"]
print(times.head())
#elapsed_time = [time - times[0] for time in times]
"""plt.plot(np.array(elapsed_time)/dt.timedelta(days=1))
plt.xlabel("Observation")
plt.ylabel("Elapsed time (days)")
plt.show()"""

# Calculating daily mean speed
"""data = bird_data[bird_data.bird_name == "Eric"]
times = data.time_stamp
elapsed_time = [time - times[0] for time in times]
elapsed_days = np.array(elapsed_time)/dt.timedelta(days=1)
next_day = 1
inds = []
daily_mean_speed = []
for (i,t) in enumerate(elapsed_days):
    if t < next_day:
        inds.append(i)
    else:
        # Compute mean speed
        daily_mean_speed.append(np.mean(data.speed_2d[inds]))
        next_day += 1
        inds = []
plt.figure()
plt.plot(daily_mean_speed)
plt.xlabel("Day")
plt.ylabel("Mean speed (m/s)")
plt.show()"""

#Using Cartopy
"""
import cartopy.crs as ccrs
import cartopy.feature as cfeature

proj = ccrs.Mercator()

plt.figure(figsize=(10, 10))
ax = plt.axes(projection=proj)
ax.set_extent((-25.0, 20.0, 52.0, 10.0))
ax.add_feature(cfeature.LAND)
ax.add_feature(cfeature.OCEAN)
ax.add_feature(cfeature.COASTLINE)
ax.add_feature(cfeature.BORDERS, linestyle=':')

for name in bird_names:
    ix = birddata["bird_name"]==name
    x, y = birddata.longitude[ix], birddata.latitude[ix]
    ax.plot(x, y, ".", transform=ccrs.Geodetic(), label=name)

plt.legend(loc="upper left")
plt.saefig("birdData.pdf")
"""
