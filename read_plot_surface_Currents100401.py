# -*- coding: utf-8 -*-
"""
Created on Thu Sep 22 09:30:19 2016

@author: Bingwei Ling
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Dec 10 15:54:43 2014
This routine can plot both the observed and modeled drifter tracks.
It has various options including how to specify start positions, how long to track, 
whether to generate animation output, etc. See Readme.
@author: Bingwei Ling
Derived from previous particle tracking work by Manning, Muse, Cui, Warren.
"""

import sys
#import pytz
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from surface_Currents_M_functions100401 import get_fvcom,draw_basemap
from matplotlib import animation
from dateutil.parser import parse
import matplotlib as mpl
from pydap.client import open_url

st_run_time = datetime.now() # Caculate execution time with en_run_time
save_dir = './Results/'
MODEL = 'GOM3'
track_way = 'forward'  
############################### Options ######################################
currents_point = pd.read_csv('28-Oct-2016_12:40currents_points.csv')
#currents_point = pd.read_csv('25-Oct-2016_16:00currents_points.csv')#28-Oct-2016_12:40currents_points
############################## Common codes ###################################
#colors = ['#99ffff','#b3ffff','#ccffff','#e6ffff','#ffffff','#ffebe6','#ffc2b3','#ff9980',
            #'#ff704d','#ff471a','#e62e00','#b32400'] #16
# Hue
colors = ['#00bfff','#00ffff','#00ffbf','#00ff80','#80ff00','#bfff00','#ffff00','#ffbf00','#ff8000','#ff4000'] 
#'#0040ff',,'#00ff40','#00ff00','#ff0000','#40ff00''#00bfff',
#colors = ['#ffebe6','#ffd6cc','#ffc2b3','#ffad99','#ff9980','#ff8566','#ff704d','#ff5c33','#ff471a','#ff3300']
#loop_length = []
cmap = mpl.colors.ListedColormap(colors)
#extract the data
currents_points = {}
spds = []; toltime = []
points = {'lats':[],'lons':[]}  # collect all points we've gained
for a in currents_point.columns[1:]:
    lb = currents_point[a]
    lbs = []
    for i in lb:           
        k = i.splitlines()[1:]
        if len(k) ==1 : continue
        modpts = dict(lon=[], lat=[], time=[], spd=[])
        for j in k:
            l=j.split()[1:]
            modpts['lat'].append(float(l[0]))
            modpts['lon'].append(float(l[1]))
            modpts['spd'].append(float(l[2]))
            ts = l[-2]+' '+l[-1]
            modpts['time'].append(ts)
            if not ts in toltime:
                toltime.append(ts)
        spds.extend(modpts['spd']); 
        '''if len(modpts['time']) > len(toltime):
            toltime = modpts['time'] #'''
        #toltime.extend(modpts['time'])
        points['lats'].extend(modpts['lat']);points['lons'].extend(modpts['lon'])
        modpts = pd.DataFrame(modpts)
        #print modpts
        lbs.append(modpts)
    currents_points[a] = lbs

#mintime = min(toltime)
#toltime = set(toltime)# ;print toltime        
crang = np.linspace(0.0,2.0, num=10, endpoint=False) #color range
norm = mpl.colors.Normalize(vmin=0.0, vmax=2.0)
#minspd = np.argmin(crang-speed)
#ax.plot(points['lons'],points['lats'])
#ax.plot(psqus[1],psqus[0])
fig = plt.figure() #figsize=(16,9)
ax = fig.add_subplot(111)
draw_basemap(ax, points)  # points is using here


levels=np.arange(-80,0,10)
ss=1
cont_range = [-3,0]
draw_parallels = 'ON'

url='http://geoport.whoi.edu/thredds/dodsC/bathy/gom03_v1_0'
dataset=open_url(url)
basemap_lat=dataset['lat']
basemap_lon=dataset['lon']
basemap_topo=dataset['topo']
minlat=min(points['lats'])
maxlat=max(points['lats'])
minlon=min(points['lons'])
maxlon=max(points['lons'])
index_minlat=int(round(np.interp(minlat,basemap_lat,range(0,basemap_lat.shape[0]))))-2
index_maxlat=int(round(np.interp(maxlat,basemap_lat,range(0,basemap_lat.shape[0]))))+2
index_minlon=int(round(np.interp(minlon,basemap_lon,range(0,basemap_lon.shape[0]))))-2
index_maxlon=int(round(np.interp(maxlon,basemap_lon,range(0,basemap_lon.shape[0]))))+2
min_index_lat=min(index_minlat,index_maxlat)
max_index_lat=max(index_minlat,index_maxlat)
min_index_lon=min(index_minlon,index_maxlon)
max_index_lon=max(index_minlon,index_maxlon)
X,Y=np.meshgrid(basemap_lon[min_index_lon:max_index_lon:ss],basemap_lat[min_index_lat:max_index_lat:ss])

#mpl.rcParams['contour.negative_linestyle'] = 'solid'


cax = fig.add_axes([0.85, 0.2, 0.02, 0.6])                    
cb1 = mpl.colorbar.ColorbarBase(cax, cmap=cmap,norm=norm,spacing='proportional',orientation='vertical')#,orientation='horizontal')
cb1.set_label('M/S')

lminlat0 = min(points['lats'])+0.01;lmaxlon0 = max(points['lons'])-0.07
lminlat1 = lminlat0; lmaxlon1 = lmaxlon0+3600*0.44704/(111111*np.cos(lminlat0*np.pi/180))
def animate(n): #del ax.collections[:]; del ax.lines[:]; ax.cla(); ax.lines.remove(line)        
    '''if track_way=='backward':
        Time = (locstart_time-timedelta(hours=n)).strftime("%d-%b-%Y %H:%M")
    else:
        Time = (locstart_time+timedelta(hours=n)).strftime("%d-%b-%Y %H:%M")#'''
    #plt.suptitle('%.f%% simulated drifters ashore\n%d days, %d m, %s'%(int(round(p)),track_days,depth,Time))
    #plt.suptitle('Current model %d' % n)
    plt.suptitle('FVCOM Surface Currents\n' + toltime[n])
    #del ax.texts[:]   
    ax.cla()
    #del ax.collections[:]
    
    CS=ax.contour(X,Y,basemap_topo.topo[min_index_lat:max_index_lat:ss,index_minlon:index_maxlon:ss],levels,cmap=plt.cm.gist_earth,linewidths=0.5)
    ax.clabel(CS,fmt='%5.0f')#, np.arange(-80,0,20), inline=1)fontsize=3,
    ax.contourf(X,Y,basemap_topo.topo[min_index_lat:max_index_lat:ss,min_index_lon:max_index_lon:ss],[0,1000],colors='gray')
    #for k in range(len(maxtime)):  
        #print k
    #for j in xrange(lcp):
    for k in currents_points:
        kl = currents_points[k]
        for j in xrange(len(kl)):
            if any(kl[j]['time']==toltime[n]):
                nu = kl[j]['time'][kl[j]['time']==toltime[n]].index[0]
                
                '''if nu<1:    
                    speed = kl[j]['spd'][nu]
                    #if n<len(lon_set[j]):
                    #,label='Depth=10m' ,markersize=4
                    ax.plot(kl[j]['lon'][:nu+1],kl[j]['lat'][:nu+1],'-',color=colors[np.argmin(abs(crang-speed))])#'''
                if nu < len(kl[j]['lon'])-1 :
                    speed = kl[j]['spd'][nu]
                    cl = colors[np.argmin(abs(crang-speed))]
                #if n<len(lon_set[j]):
                    #ax.plot(kl[j]['lon'][nu:nu+2],kl[j]['lat'][nu:nu+2],'-',color=cl)
                    dx=kl[j]['lon'][nu+1]-kl[j]['lon'][nu]; dy=kl[j]['lat'][nu+1]-kl[j]['lat'][nu]
                    ax.arrow(kl[j]['lon'][nu],kl[j]['lat'][nu],dx,dy,fc=cl, ec=cl,head_width=0.005, head_length=0.01)
                    #rm.append(aw)
                #ax.scatter(kl[j]['lon'][nu],kl[j]['lat'][nu],s=2,c=cl)
                #ax.plot(kl[j]['lon'][nu],kl[j]['lat'][nu],'o',markeredgecolor=cl,markerfacecolor=cl,markersize=35)   
                '''x=kl[j]['lon'][nu-1];y=kl[j]['lat'][nu-1]
                dx=kl[j]['lon'][nu]-x;dy=kl[j]['lat'][nu]-y
                ax.arrow(x,y,dx,dy, fc=cl, ec=cl)##'''
    #ax.plot([lmaxlon0,lmaxlon1],[lminlat0,lminlat1],'-|',color='k',linewidth=3)
    #ax.text(lmaxlon1,lminlat1+0.004,'1 mph')
    #la=plt.colorbar(ax=ax)
    #la.set_label('Model Water Depth (m)', rotation=-90)
anim = animation.FuncAnimation(fig, animate, frames=len(toltime),interval=500) #,
en_run_time = datetime.now()
print 'Take '+str(en_run_time-st_run_time)+' running the code.\nStart at '+str(st_run_time)+'\nEnd at   '+str(en_run_time)
#print 'Min-spd,max-spd',crang[0],crang[-1]mencoder
anim.save(save_dir+'%s-%s_%s.gif'%(MODEL,track_way,en_run_time.strftime("%d-%b-%Y_%H:%M")),writer='imagemagick',dpi=400) #,,,fps=1
plt.show()
