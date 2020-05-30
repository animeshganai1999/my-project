import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import cm
import matplotlib.colors as col
from matplotlib.backend_bases import MouseButton
np.random.seed(12345)
df = pd.DataFrame([np.random.normal(32000,200000,3650), 
                   np.random.normal(43000,100000,3650), 
                   np.random.normal(43500,140000,3650), 
                   np.random.normal(48000,70000,3650)], 
                  index=[1992,1993,1994,1995])
df = df.T
plt.figure(figsize = (6,9))
df1 = df.describe()
df1 = df1.T

'''statistics part
calculating 95% confidence boundary for the mean'''
df1['yerr'] = 1.96*df1['std'] / np.sqrt(df1['count'])
df1['lower'] = df1['mean'] - df1['yerr']
df1['upper'] = df1['mean'] + df1['yerr']
'''end of calculation'''
df1 = df1.T
x = np.linspace(0,3,4)

#Computing Probability for putting colors in bars
def compute_probability(y):
    global df1
    result = []
    for i in range(len(df1.T)):
        v = i
        if y > df1.loc['upper'].iloc[v]:
            result.append('blue')
        elif y < df1.loc['lower'].iloc[v]:
            result.append('red')
        else:
            result.append('#AAA5A5')
    return result

#Defining colors for each bars


ax = plt.gca()
ax.set_ylim(0,60000)
plt.xticks(x,['1992','1993','1994','1995'])
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

#colors = compute_probability(y)

bars = plt.bar(x,df1.loc['mean'],yerr = df1.loc['yerr'],width = 0.8,
               color = 'blue',edgecolor = 'black')
hline = ax.axhline(60000,color = 'green', zorder=1)
def mouse_move(event):
    if not event.inaxes:
        return
    y = event.ydata
    global bars   
    global hline    
    colors = compute_probability(y)
    hline.set_ydata(y)
    for i in range(len(bars)):
        bars[i].set_color(colors[i])
    plt.draw()

def on_click(event):
    if event.button is MouseButton.LEFT:
        print("Disconnecting Call Back")
        plt.disconnect(binding_id)
    
#cursor moving event
binding_id = plt.connect('motion_notify_event', mouse_move)
plt.connect('button_press_event',on_click)
plt.title("Easiest Option")
fig = plt.gcf()
cmap = cm.get_cmap('coolwarm')
cpick = cm.ScalarMappable(cmap=cmap, norm=col.Normalize(vmin=0, vmax=1.0))
cpick.set_array([])
cbar = plt.colorbar(cpick, orientation="horizontal")

plt.show()

