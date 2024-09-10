import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker 

PATH = ''
df1 = pd.read_csv(PATH + 'czas_adnotacji_jedna_zasada.csv')
df2 = pd.read_csv(PATH + 'czas_adnotacji_dwie_zasady.csv')
df3 = pd.read_csv(PATH + 'czas_adnotacji_trzy_zasady.csv')

df = pd.concat([df1, df2, df3])

print(df.head())

print(df.describe())

df.sort_values(by=['liczba porownan iou'], inplace=True)
df['liczba porownan iou_2'] = df['liczba porownan iou']/1000000
fig, ax = plt.subplots()
#df.plot(x='liczba porownan iou', y='czas [s]')
#ax.ticklabel_format(style='plain')
plt.plot(df['liczba porownan iou_2'], df['czas [s]'])
plt.xlabel("Number of calculated IoU values [mln]")
plt.ylabel("Operating time [s]")
ax.xaxis.set_major_locator(ticker.MultipleLocator(0.5))
plt.title("Operating time of algorithm")
plt.savefig(f"czas_działania.png")
