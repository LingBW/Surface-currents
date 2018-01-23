# -*- coding: utf-8 -*-
"""
Created on Thu Sep 22 09:30:19 2016
Surface currents,depth contour and drifter forecast tracks.Create at 3 Nov,2016.
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
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from surface_Currents_M_functions08152017 import get_fvcom,draw_basemap,get_drifter
from matplotlib import animation
import matplotlib as mpl
from pydap.client import open_url

st_run_time = datetime.now() # Caculate execution time with en_run_time
         
########################## Hard codes ############################################
MODEL = 'GOM3'     # 'ROMS', 'GOM3','massbay','30yr'
GRIDS = ['GOM3','massbay','30yr']    # All belong to FVCOM. '30yr' works from 1977/12/31 22:58 to 2014/1/1 0:0
#depth = 1    # depth below ocean surface, positive
track_hours = 24    #MODEL track time(days)
track_way = 'forward'    # Three options: backward, forward and both. 'both' only apply to Option 2 and 3.
image_style = 'plot'      # Two option: 'plot', animation
# You can track form now by specify start_time = datetime.now(pytz.UTC) 
#start_time = datetime(2013,10,19,12,0,0,0)#datetime.now(pytz.UTC) 
#start_time = datetime.utcnow()
#end_time = start_time + timedelta(hours=track_hours)

model_boundary_switch = 'ON' # OFF or ON. Only apply to FVCOM
streamline = 'OFF'
wind = 'OFF'
bcon = 'stop' #boundary-condition: stop,reflection

save_dir = './Results/'
#colors = ['magenta','cyan','olive','blue','orange','green','red','yellow','black','purple']
#'#0039b3','#0049e6','#1a62ff','#4d85ff','#80a8ff',,'#b3cbff','#e6eeff'
#colors = ['#99ffff','#b3ffff','#ccffff','#e6ffff','#ffffff','#ffebe6','#ffc2b3','#ff9980','#ff704d','#ff471a','#e62e00','#b32400'] #16
'''
utcti = datetime.utcnow(); utct = utcti.strftime('%H')
locti = datetime.now(); loct = locti.strftime('%H')
ditnu = int(utct)-int(loct) # the deference between UTC and local time .
if ditnu < 0:
    ditnu = int(utct)+24-int(loct)
locstart_time = start_time - timedelta(hours=ditnu)#'''

############################## Drifter parts ##################################
drifterIDs = ['176440663']#,'1604107014''160410704',,'160410707'
clrs = ['b','g','m','y','c']
INPUT_DATA = 'drift_X.dat'
track_days = 2 #the days of real drifter track.
dstart_time = datetime.utcnow()-timedelta(track_days)

print 'Get drifter_points'
drifter_points = {} #dict(lon=[],lat=[])
start_times = []
cenlat,cenlon = [],[]
for i in drifterIDs:
    dps = []; 
    drifter = get_drifter(i, INPUT_DATA)
    dr_points = drifter.get_track(dstart_time,track_days)
    dps.append(dr_points['lon']); dps.append(dr_points['lat'])
    start_times.append(dr_points['time'][-1])
    cenlon.append(dr_points['lon'][-1]); cenlat.append(dr_points['lat'][-1]);
    drifter_points[i]=dps
    
start_time = min(start_times)-timedelta(track_days) # datetime.utcnow()
#print 'start_time',start_time
end_time = start_time + timedelta(hours=track_hours)

#print "cenlat,cenlon,np.mean(cenlat),np.mean(cenlon)",cenlat,cenlon,'n/',np.mean(cenlat),np.mean(cenlon)
centerpoint = (np.mean(cenlat),np.mean(cenlon))
bordersidele = 0.5
#centerpoint = (41.91,-70.233)
#bordersidele = 0.22
############################## model codes ###################################
print 'Get model_points'
points = {'lats':[],'lons':[]}  # collect all points we've gained   
model_points = {};
get_obj = get_fvcom(MODEL)
toltime = get_obj.get_url(start_time,end_time)
b_points,psqus = get_obj.get_data(centerpoint,bordersidele)# b_points is model boundary points.
points['lons'].extend(psqus[1]);points['lats'].extend(psqus[0])

for j in range(len(drifterIDs)):
    dlon = drifter_points[drifterIDs[j]][0][-1]
    dlat = drifter_points[drifterIDs[j]][1][-1]
    mdp = get_obj.get_dtrack(start_times[j],dlon,dlat,track_way)
    model_points[drifterIDs[j]] = mdp
# Core codes
currents_points = {}
print 'Get currents_points'
#for j in range(track_hours):#len(toltime)
    #if j==0 or (j+1)%2==0 :
    #if j%6==0:
currents_points = get_obj.current_track_new(track_way)
#currents_points[toltime[j]] = cpoints#'''
spds = []
for k in currents_points:
    kl = currents_points[k]#; cl = []; sd = []
    for j in kl:
        #j {'5656':[[(),()],[(),()]]}
        for l in j.values()[0]:
            
            dx = l[1][0]-l[0][0]; dy = l[1][1]-l[0][1]
            pspeed = math.sqrt(dx**2+dy**2)
            spds.append(pspeed)
        '''if len(cl)<300:
            cl.append(kl[j]); sd.append(pspeed)
        else:
            ix = np.argmin(sd)
            if pspeed > sd[ix]:
                cl[ix] = kl[j]; sd[ix] = pspeed
    currents_points[k] = cl; print len(cl)#'''
crang = np.linspace(min(spds),max(spds), num=10) #color range
'''try:
    pd.DataFrame(currents_points).to_csv(st_run_time.strftime("%d-%b-%Y_%H:%M")+'currents_points.csv')
    #np.save('currents_points',np.array(currents_points))
except:
    print 'Failed to save the data to a file.'
    pass#'''
#lcp = len(currents_points)

#loop_length = []
fig = plt.figure() #figsize=(16,9)
ax = fig.add_subplot(111)
#draw_basemap(ax, points)  # points is using here

########################### get depth contour data####################################
'''levels=np.arange(-80,0,10)
ss=1
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
#'''

####################### setting scale#############################################
#colors = ['#00bfff','#00ffff','#00ffbf','#00ff80','#80ff00','#bfff00','#ffff00','#ffbf00','#ff8000','#ff4000']
#colors = ['#e6f2ff','#cce6ff','#b3d9ff','#99ccff','#80bfff','#66b3ff','#4da6ff','#3399ff','#1a8cff','#0080ff'] # blue
colors = ['#cce6ff','#99ccff','#66b3ff','#33cccc','#ccffcc','#ccff99','#ffffcc','#ffcc99','#ff9966','#ff6600']
cont_range = [-3,0]
draw_parallels = 'ON'
cmap = mpl.colors.ListedColormap(colors)
norm = mpl.colors.Normalize(vmin=0, vmax=2.0)
cax = fig.add_axes([0.85, 0.2, 0.02, 0.6])                    
cb1 = mpl.colorbar.ColorbarBase(cax, cmap=cmap,norm=norm,spacing='proportional',orientation='vertical')#,orientation='horizontal')
cb1.set_label('M/S')#'''

lminlat0 = min(points['lats'])+0.01;lmaxlon0 = max(points['lons'])-0.07
lminlat1 = lminlat0; lmaxlon1 = lmaxlon0+1609.344/(111111*np.cos(lminlat0*np.pi/180)) # 1 mile = 1609.344 meter'''

def animate(n): #del ax.collections[:]; del ax.lines[:]; ax.cla(); ax.lines.remove(line)        
    '''if track_way=='backward':
        Time = (locstart_time-timedelta(hours=n)).strftime("%d-%b-%Y %H:%M")
    else:
        Time = (locstart_time+timedelta(hours=n)).strftime("%d-%b-%Y %H:%M")#'''
    #plt.suptitle('%.f%% simulated drifters ashore\n%d days, %d m, %s'%(int(round(p)),track_days,depth,Time))
    #plt.suptitle('Current model %d' % n)
    ax.cla()
    #del ax.collections[:]
    #del ax.lines[:]
    draw_basemap(ax, points)  # points is using her
    plt.suptitle('FVCOM Surface Currents \n' + toltime[n].strftime("%d-%b-%Y %H:%M"))
    #del ax.lines[:] 
    for i in range(len(drifterIDs)):
        lons = drifter_points[drifterIDs[i]][0]; lats = drifter_points[drifterIDs[i]][1];
        ax.plot(lons,lats,'-',color=clrs[i],lw=3.)
        #if any(model_points[i]['time']==toltime[n]):
        if toltime[n] in model_points[drifterIDs[i]]['time']:
            #du = model_points[i]['time'][model_points[i]['time']==toltime[n]].index[0]
            du = model_points[drifterIDs[i]]['time'].index(toltime[n])
            ax.plot(model_points[drifterIDs[i]]['lon'][:du+1],model_points[drifterIDs[i]]['lat'][:du+1],'-',color='r',lw=3.)
        if toltime[n] > model_points[drifterIDs[i]]['time'][-1]:
            ax.plot(model_points[drifterIDs[i]]['lon'],model_points[drifterIDs[i]]['lat'],'-',color='r',lw=3.)#'''
    '''# plot depth contour###########################
    CS=ax.contour(X,Y,basemap_topo.topo[min_index_lat:max_index_lat:ss,index_minlon:index_maxlon:ss],levels,cmap=plt.cm.gist_earth,linewidths=0.5)
    ax.clabel(CS,fmt='%5.0f')#, np.arange(-80,0,20), inline=1)fontsize=3,
    ax.contourf(X,Y,basemap_topo.topo[min_index_lat:max_index_lat:ss,min_index_lon:max_index_lon:ss],[0,1000],colors='gray')#'''
    
    #for k in currents_points:
    kl = currents_points[toltime[n]]
    for j in kl:
        '''# scatter plotting
        pp = np.vstack(kl[j]).T
        ax.scatter(pp[0],pp[1],c=['b','g','y','r'],s=[10,20,30])#'''
        for l in j.values()[0]:
            dx = l[1][0]-l[0][0]; dy = l[1][1]-l[0][1]
            speed = math.sqrt(dx**2+dy**2)
            cl=colors[np.argmin(abs(crang-speed))]
            ax.arrow(l[0][0],l[0][1],dx,dy,fc=cl,ec=cl,head_width=0.005, head_length=0.01)#'''
            
    # plot the scale
    ax.plot([lmaxlon0,lmaxlon1],[lminlat0,lminlat1],'-|',color='k',linewidth=3)
    ax.text(lmaxlon1,lminlat1+0.004,'1 mph')#'''
anim = animation.FuncAnimation(fig, animate, frames=track_hours,interval=500) #,
en_run_time = datetime.now()
print 'Take '+str(en_run_time-st_run_time)+' running the code.\nStart at '+str(st_run_time)+'\nEnd at   '+str(en_run_time)
#print 'Min-spd,max-spd',crang[0],crang[-1]
anim.save(save_dir+'%s-%s_%s.gif'%(MODEL,track_way,en_run_time.strftime("%d-%b-%Y_%H:%M")),writer='imagemagick',dpi=250) #,,,fps=1

plt.show()
