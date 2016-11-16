# -*- coding: utf-8 -*-
"""
Created on Thu Sep 22 11:33:53 2016

@author: Bingwei Ling
"""

import sys
import netCDF4
#import calendar
import matplotlib.pyplot as plt
from datetime import datetime,timedelta
import numpy as np
import pandas as pd
from dateutil.parser import parse
#import pytz
from matplotlib.path import Path
import math
from mpl_toolkits.basemap import Basemap
import colorsys
from sympy import *
from sympy.geometry import *
from fractions import Fraction


def get_nc_data(url, *args):
    '''
    get specific dataset from url

    *args: dataset name, composed by strings
    ----------------------------------------
    example:
        url = 'http://www.nefsc.noaa.gov/drifter/drift_tcs_2013_1.dat'
        data = get_url_data(url, 'u', 'v')
    '''
    nc = netCDF4.Dataset(url)
    data = {}
    for arg in args:
        try:
            data[arg] = nc.variables[arg]
        except (IndexError, NameError, KeyError):
            print 'Dataset {0} is not found'.format(arg)
    return data
    
class get_fvcom():
    
    def __init__(self, mod):
        self.modelname = mod
    def points_square(self,point, hside_length):
        '''point = (lat,lon); length: units is decimal degrees.
           return a squre points(lats,lons) on center point,without center point'''
        ps = []
        (lat,lon) = point; 
        length =float(hside_length)
        #lats=[lat]; lons=[lon]
        #lats=[]; lons=[]
        bbox = [lon-length, lon+length, lat-length, lat+length]
        bbox = np.array(bbox)
        self.points = np.array([bbox[[0,1,1,0]],bbox[[2,2,3,3]]])
        #print points
        pointt = self.points.T
        for i in pointt:
            ps.append((i[1],i[0]))
        ps.append((pointt[0][1],pointt[0][0]))# add first point one more time for Path.
        #lats.extend(points[1]); lons.extend(points[0])
        #bps = np.vstack((lon,lat)).T
        #return lats,lons
        return ps
        
    def nearest_point(self, lon, lat, lons, lats, length):  #0.3/5==0.06
        '''Find the nearest point to (lon,lat) from (lons,lats),
           return the nearest-point (lon,lat)
           author: Bingwei'''
        p = Path.circle((lon,lat),radius=length)
        #numpy.vstack(tup):Stack arrays in sequence vertically
        points = np.vstack((lons.flatten(),lats.flatten())).T  
        
        insidep = []
        #collect the points included in Path.
        for i in xrange(len(points)):
            if p.contains_point(points[i]):# .contains_point return 0 or 1
                insidep.append(points[i])  
        # if insidep is null, there is no point in the path.
        if not insidep:
            print 'There is no model-point near the given-point.'
            raise Exception()
        #calculate the distance of every points in insidep to (lon,lat)
        distancelist = []
        for i in insidep:
            ss=math.sqrt((lon-i[0])**2+(lat-i[1])**2)
            distancelist.append(ss)
        # find index of the min-distance
        mindex = np.argmin(distancelist)
        # location the point
        lonp = insidep[mindex][0]; latp = insidep[mindex][1]
        
        return lonp,latp
        
    def get_url(self, starttime, endtime):
        '''
        get different url according to starttime and endtime.
        urls are monthly.
        '''
        self.hours = int(round((endtime-starttime).total_seconds()/60/60))
        #print self.hours
                
        if self.modelname == "GOM3":
            turl = '''http://www.smast.umassd.edu:8080/thredds/dodsC/FVCOM/NECOFS/Forecasts/NECOFS_GOM3_FORECAST.nc'''
            
            try:
                self.tdata = netCDF4.Dataset(turl).variables
                #MTime = tdata['time'] 
                MTime = self.tdata['Times']
            except:
                print '"GOM3" database is unavailable!'
                raise Exception()
            #Times = netCDF4.num2date(MTime[:],MTime.units)
            Times = []
            for i in MTime:
                strt = '201'+i[3]+'-'+i[5]+i[6]+'-'+i[8]+i[9]+' '+i[11]+i[12]+':'+i[14]+i[15]
                Times.append(datetime.strptime(strt,'%Y-%m-%d %H:%M'))#'''
            fmodtime = Times[0]; emodtime = Times[-1]         
            if starttime<fmodtime or starttime>emodtime or endtime<fmodtime or endtime>emodtime:
                print 'Time: Error! Model(GOM3) only works between %s with %s(UTC).'%(fmodtime,emodtime)
                raise Exception()
            npTimes = np.array(Times)
            tm1 = npTimes-starttime; #tm2 = mtime-t2
            self.index1 = np.argmin(abs(tm1))#'''
            #index1 = netCDF4.date2index(starttime,MTime,select='nearest')
            self.index2 = self.index1 + self.hours#'''
            #print 'index1,index2',index1,index2
            #url = url.format(index1, index2)
            self.mTime = Times[self.index1:self.index2+1]
            
            self.url = turl
            
        elif self.modelname == "massbay":
            turl = '''http://www.smast.umassd.edu:8080/thredds/dodsC/FVCOM/NECOFS/Forecasts/NECOFS_FVCOM_OCEAN_MASSBAY_FORECAST.nc?'''
            
            try:
                self.tdata = netCDF4.Dataset(turl).variables
                #MTime = tdata['time'] 
                MTime = self.tdata['Times']              
            except:
                print '"massbay" database is unavailable!'
                raise Exception()
            Times = []
            for i in MTime:
                strt = '201'+i[3]+'-'+i[5]+i[6]+'-'+i[8]+i[9]+' '+i[11]+i[12]+':'+i[14]+i[15]
                Times.append(datetime.strptime(strt,'%Y-%m-%d %H:%M'))
            fmodtime = Times[0]; emodtime = Times[-1]         
            if starttime<fmodtime or starttime>emodtime or endtime<fmodtime or endtime>emodtime:
                print 'Time: Error! Model(massbay) only works between %s with %s(UTC).'%(fmodtime,emodtime)
                raise Exception()
            npTimes = np.array(Times)
            tm1 = npTimes-starttime; #tm2 = mtime-t2
            index1 = np.argmin(abs(tm1))
            index2 = index1 + self.hours#'''
            #url = url.format(index1, index2)
            self.mTime = Times[index1:index2+1]
            
            self.url = turl

        elif self.modelname == "30yr": #start at 1977/12/31 23:00, end at 2014/1/1 0:0, time units:hours
            timeurl = """http://www.smast.umassd.edu:8080/thredds/dodsC/fvcom/hindcasts/30yr_gom3?time[0:1:316008]"""
            url = '''http://www.smast.umassd.edu:8080/thredds/dodsC/fvcom/hindcasts/30yr_gom3?h[0:1:48450],
            lat[0:1:48450],latc[0:1:90414],lon[0:1:48450],lonc[0:1:90414],nbe[0:1:2][0:1:90414],siglay[0:1:44][0:1:48450],
            u[{0}:1:{1}][0:1:44][0:1:90414],v[{0}:1:{1}][0:1:44][0:1:90414],zeta[{0}:1:{1}][0:1:48450]'''
            
            try:
                mtime = netCDF4.Dataset(timeurl).variables['time'][:]
            except:
                print '"30yr" database is unavailable!'
                raise Exception
            # get model's time horizon(UTC).
            '''fmodtime = datetime(1858,11,17) + timedelta(float(mtime[0]))
            emodtime = datetime(1858,11,17) + timedelta(float(mtime[-1]))
            mstt = fmodtime.strftime('%m/%d/%Y %H:%M')
            mett = emodtime.strftime('%m/%d/%Y %H:%M') #'''
            # get number of days from 11/17/1858
            t1 = (starttime - datetime(1858,11,17)).total_seconds()/86400 
            t2 = (endtime - datetime(1858,11,17)).total_seconds()/86400
            if not mtime[0]<t1<mtime[-1] or not mtime[0]<t2<mtime[-1]:
                #print 'Time: Error! Model(massbay) only works between %s with %s(UTC).'%(mstt,mett)
                print 'Time: Error! Model(massbay) only works between 1978-1-1 with 2014-1-1(UTC).'
                raise Exception()
            
            tm1 = mtime-t1; #tm2 = mtime-t2
            index1 = np.argmin(abs(tm1)); #index2 = np.argmin(abs(tm2)); print index1,index2
            index2 = index1 + self.hours
            url = url.format(index1, index2)
            Times = []
            for i in range(self.hours+1):
                Times.append(starttime+timedelta(hours=i))
            self.mTime = Times
            self.url = url
        #print url
        return self.mTime
        
    def get_data(self,point,leh):
        '''
        "get_data" not only returns boundary points but defines global attributes to the object
        '''
        #self.data = get_nc_data(url,'lat','lon','latc','lonc','siglay','h','nbe','u','v','zeta')#,'nv'
        self.lonc, self.latc = self.tdata['lonc'][:], self.tdata['latc'][:]  #quantity:165095
        #self.lons, self.lats = self.data['lon'][:], self.data['lat'][:]
        #self.h = self.data['h'][:]; self.siglay = self.data['siglay'][:]; #nv = self.data['nv'][:]
        self.u = self.tdata['u'][self.index1:self.index2,0,:][:]; self.v = self.tdata['v'][self.index1:self.index2,0,:][:]#; self.zeta = self.data['zeta']
        print 'model length',len(self.u),len(self.v)
        nbe1=self.tdata['nbe'][0];nbe2=self.tdata['nbe'][1];
        nbe3=self.tdata['nbe'][2]
        pointt = np.vstack((nbe1,nbe2,nbe3)).T; self.pointt = pointt
        wl=[]
        for i in pointt:
            if 0 in i: 
                wl.append(1)
            else:
                wl.append(0)
        self.wl = wl
        tf = np.array(wl)
        self.inde = np.where(tf==True)
        #b_index = inde[0]
        lonb = self.lonc[self.inde]; latb = self.latc[self.inde]        
        self.b_points = np.vstack((lonb,latb)).T#'''
        #self.b_points = b_points
        (lat,lon) = point
        self.lonl,self.latl = self.shrink_data(lon,lat,self.lonc,self.latc,leh)
        
        return self.b_points,self.points #,nv lons,lats,lonc,latc,,h,siglay
        
    def shrink_data(self,lon,lat,lons,lats,le):
        lont = []; latt = []
        #p = Path.circle((lon,lat),radius=rad)
        self.psqus = self.points_square((lon,lat),le) # Got four point of rectangle with center point (lon,lat)
        codes = [Path.MOVETO,Path.LINETO,Path.LINETO,Path.LINETO,Path.CLOSEPOLY,]
        #print psqus
        self.sp = Path(self.psqus,codes)
        pints = np.vstack((lons,lats)).T
        for i in range(len(pints)):
            if self.sp.contains_point(pints[i]):
                lont.append(pints[i][0])
                latt.append(pints[i][1])
        lonl=np.array(lont); latl=np.array(latt)#'''
        if not lont:
            print 'Given point out of model area.'
            sys.exit()
        return lonl,latl
        
    def eline_path(self,lon,lat):
        '''
        When drifter close to boundary(less than 0.1),find one nearest point to drifter from boundary points, 
        then find two nearest boundary points to previous boundary point, create a boundary path using that 
        three boundary points.
        '''
        def boundary_location(locindex,pointt,wl):
            '''
            Return the index of boundary points nearest to 'locindex'.
            '''
            loca = []
            dx = pointt[locindex]; #print 'func',dx 
            for i in dx: # i is a number.
                #print i  
                if i ==0 :
                    continue
                dx1 = pointt[i-1]; #print dx1
                if 0 in dx1:
                    loca.append(i-1)
                else:
                    for j in dx1:
                        if j != locindex+1:
                            if wl[j-1] == 1:
                                loca.append(j-1)
            return loca
        
        p = Path.circle((lon,lat),radius=0.02) #0.06
        dis = []; bps = []; pa = []
        tlons = []; tlats = []; loca = []
        for i in self.b_points:
            if p.contains_point(i):
                bps.append((i[0],i[1]))
                d = math.sqrt((lon-i[0])**2+(lat-i[1])**2)
                dis.append(d)
        bps = np.array(bps)
        if not dis:
            return None
        else:
            print "Close to boundary."
            dnp = np.array(dis)
            dmin = np.argmin(dnp)
            lonp = bps[dmin][0]; latp = bps[dmin][1]
            index1 = np.where(self.lonc==lonp)
            index2 = np.where(self.latc==latp)
            elementindex = np.intersect1d(index1,index2)[0] # location 753'''
            #print index1,index2,elementindex  
            loc1 = boundary_location(elementindex,self.pointt,self.wl) ; #print 'loc1',loc1
            loca.extend(loc1)
            loca.insert(1,elementindex)               
            for i in range(len(loc1)):
                loc2 = boundary_location(loc1[i],self.pointt,self.wl); #print 'loc2',loc2
                if len(loc2)==1:
                    continue
                for j in loc2:
                    if j != elementindex:
                        if i ==0:
                            loca.insert(0,j)
                        else:
                            loca.append(j)
            
            for i in loca:
                tlons.append(self.lonc[i]); tlats.append(self.latc[i])
                       
            for i in xrange(len(tlons)):
                pa.append((tlons[i],tlats[i]))
            #path = Path(pa)#,codes
            return pa
    def current_track(self,jn,track_way): #,point,leh):
        cts = []        
        #self.lonk,self.latk = self.shrink_data(lon,lat,self.lons,self.lats,leh)
        epoints = np.vstack((self.lonl[::3],self.latl[::3])).T
        numep = len(epoints)
        for i in range(numep):
            #print '%d of %d, %d' % (i+1,numep,jn+1)
            getk = self.get_track(jn,epoints[i][0],epoints[i][1],track_way)
            #print type(getk['lon']),type(getk['lat']),type(getk['layer']),type(getk['spd'])
            ld = min(len(getk['lon']),len(getk['lat']),len(getk['spd']))
            for j in getk:
                if len(getk[j])>ld:
                    getk[j] = getk[j][:ld]
            #print getk
            pgetk = pd.DataFrame(getk)
            
            #print pgetk
            cts.append(pgetk)
        return cts
        
    def get_track(self,jnu,lon,lat,track_way): #,b_index,nvdepth,,bcon 
        '''
        Get forecast points start at lon,lat
        '''
        modpts = dict(lon=[lon], lat=[lat], time=[], spd=[]) #model forecast points, layer=[]
        
        try:
            if self.modelname == "GOM3" or self.modelname == "30yr":
                lonp,latp = self.nearest_point(lon, lat, self.lonl, self.latl,0.2)
                #lonn,latn = self.nearest_point(lon,lat,self.lonk,self.latk,0.3)
            if self.modelname == "massbay":
                lonp,latp = self.nearest_point(lon, lat, self.lonl, self.latl,0.03)
                #lonn,latn = self.nearest_point(lon,lat,self.lonk,self.latk,0.05)        
            index1 = np.where(self.lonc==lonp)
            index2 = np.where(self.latc==latp)
            elementindex = np.intersect1d(index1,index2)
            
        except:
            return modpts#,0  
            
        t = abs(self.hours) 
    
        for i in xrange(t):            
            '''if i!=0 and i%24==0 :
                #print 'layer,lon,lat,i',layer,lon,lat,i
                lonl,latl = self.shrink_data(lon,lat,self.lonc,self.latc,0.5)
                lonk,latk = self.shrink_data(lon,lat,self.lons,self.lats,0.5)#'''
            if i<jnu: continue
            
            if track_way=='backward':
                u_t1 = -self.u[t-1-i,elementindex][0]; v_t1 = -self.v[t-1-i,elementindex][0]
                modpts['time'].append(self.mTime[t-1-i])
            else:
                u_t1 = self.u[i,elementindex][0]; v_t1 = self.v[i,elementindex][0]
                modpts['time'].append(self.mTime[i])
                #u_t2 = self.u[i+1,layer,elementindex][0]; v_t2 = self.v[i+1,layer,elementindex][0]
            #u_t,v_t = self.uvt(u_t1,v_t1,u_t2,v_t2)
            #u_t = (u_t1+u_t2)/2; v_t = (v_t1+v_t2)/2
            
            dx = 60*60*u_t1; dy = 60*60*v_t1
            pspeed = math.sqrt(u_t1**2+v_t1**2)
            modpts['spd'].append(pspeed)
            if i == t-1:# stop when got the last point speed.
                return modpts#,2
            if i >= jnu+4:# break
                return modpts#,2
            #x,y = mapx(lon,lat)
            #temlon,temlat = mapx(x+dx,y+dy,inverse=True)            
            temlon = lon + (dx/(111111*np.cos(lat*np.pi/180)))
            temlat = lat + dy/111111 #'''
            
            #########################
            lon = temlon; lat = temlat
            modpts['lon'].append(lon); modpts['lat'].append(lat)#; modpts['layer'].append(layer); 
            #print '%d,Lat,Lon,Speed'%(i+1),temlat,temlon,pspeed              
            try:
                if self.modelname == "GOM3" or self.modelname == "30yr":
                    lonp,latp = self.nearest_point(lon, lat, self.lonl, self.latl,0.2)
                    #lonn,latn = self.nearest_point(lon,lat,self.lonk,self.latk,0.3)
                if self.modelname == "massbay":
                    lonp,latp = self.nearest_point(lon, lat, self.lonl, self.latl,0.03)
                    #lonn,latn = self.nearest_point(lon,lat,self.lonk,self.latk,0.05)
                index1 = np.where(self.lonc==lonp)
                index2 = np.where(self.latc==latp)
                elementindex = np.intersect1d(index1,index2); #print 'elementindex',elementindex
                #index3 = np.where(self.lons==lonn)
                #index4 = np.where(self.lats==latn)
                #nodeindex = np.intersect1d(index3,index4)
                if elementindex in self.inde[0]:
                    #print 'boundary elementindex',elementindex,type(elementindex)
                    return modpts#,2  # hits the boundary.
            except:
                return modpts#,1
    def get_dtrack(self,dtime,lon,lat,track_way): #,b_index,nvdepth,,bcon 
        '''
        Get forecast points start at lon,lat
        '''
        modpts = dict(lon=[lon], lat=[lat], time=[], spd=[]) #model forecast points, layer=[]
        tm1 = np.array(self.mTime)-dtime
        jnu = np.argmin(abs(tm1))
        try:
            if self.modelname == "GOM3" or self.modelname == "30yr":
                lonp,latp = self.nearest_point(lon, lat, self.lonl, self.latl,0.2)
                #lonn,latn = self.nearest_point(lon,lat,self.lonk,self.latk,0.3)
            if self.modelname == "massbay":
                lonp,latp = self.nearest_point(lon, lat, self.lonl, self.latl,0.03)
                #lonn,latn = self.nearest_point(lon,lat,self.lonk,self.latk,0.05)        
            index1 = np.where(self.lonc==lonp)
            index2 = np.where(self.latc==latp)
            elementindex = np.intersect1d(index1,index2)
            
        except:
            return modpts#,0  
            
        t = abs(self.hours) 
    
        for i in xrange(t):            
            '''if i!=0 and i%24==0 :
                #print 'layer,lon,lat,i',layer,lon,lat,i
                lonl,latl = self.shrink_data(lon,lat,self.lonc,self.latc,0.5)
                lonk,latk = self.shrink_data(lon,lat,self.lons,self.lats,0.5)#'''
            if i<jnu: continue
            #modpts['time'].append(self.mTime[i])
            #u_t1 = self.u[i,elementindex][0]; v_t1 = self.v[i,elementindex][0]
            if track_way=='backward':
                u_t1 = -self.u[t-1-i,elementindex][0]; v_t1 = -self.v[t-1-i,elementindex][0]
                modpts['time'].append(self.mTime[t-1-i])
            else:
                u_t1 = self.u[i,elementindex][0]; v_t1 = self.v[i,elementindex][0]
                modpts['time'].append(self.mTime[i])
                #u_t2 = self.u[i+1,layer,elementindex][0]; v_t2 = self.v[i+1,layer,elementindex][0]
            #u_t,v_t = self.uvt(u_t1,v_t1,u_t2,v_t2)
            #u_t = (u_t1+u_t2)/2; v_t = (v_t1+v_t2)/2
            
            dx = 60*60*u_t1; dy = 60*60*v_t1
            pspeed = math.sqrt(u_t1**2+v_t1**2)
            modpts['spd'].append(pspeed)
            if i == t-1:# stop when got the last point speed.
                return modpts#,2
            #if i >= jnu+4:# break
                #return modpts#,2
            #x,y = mapx(lon,lat)
            #temlon,temlat = mapx(x+dx,y+dy,inverse=True)            
            temlon = lon + (dx/(111111*np.cos(lat*np.pi/180)))
            temlat = lat + dy/111111 #'''
            
            #########################
            lon = temlon; lat = temlat
            modpts['lon'].append(lon); modpts['lat'].append(lat)#; modpts['layer'].append(layer); 
            #print '%d,Lat,Lon,Speed'%(i+1),temlat,temlon,pspeed              
            try:
                if self.modelname == "GOM3" or self.modelname == "30yr":
                    lonp,latp = self.nearest_point(lon, lat, self.lonl, self.latl,0.2)
                    #lonn,latn = self.nearest_point(lon,lat,self.lonk,self.latk,0.3)
                if self.modelname == "massbay":
                    lonp,latp = self.nearest_point(lon, lat, self.lonl, self.latl,0.03)
                    #lonn,latn = self.nearest_point(lon,lat,self.lonk,self.latk,0.05)
                index1 = np.where(self.lonc==lonp)
                index2 = np.where(self.latc==latp)
                elementindex = np.intersect1d(index1,index2); #print 'elementindex',elementindex
                #index3 = np.where(self.lons==lonn)
                #index4 = np.where(self.lats==latn)
                #nodeindex = np.intersect1d(index3,index4)
                if elementindex in self.inde[0]:
                    print 'boundary elementindex',elementindex,type(elementindex)
                    return modpts#,2  # hits the boundary.
            except:
                return modpts#,1                              
        #return modpts,2
def draw_basemap(ax, points, interval_lon=0.5, interval_lat=0.5):
    '''
    draw the basemap?
    '''
    
    lons = points['lons']
    lats = points['lats']
    #size = max((max(lons)-min(lons)),(max(lats)-min(lats)))/2
    size = 0
    map_lon = [min(lons)-size,max(lons)+size]
    map_lat = [min(lats)-size,max(lats)+size]
    
    #ax = fig.sca(ax)
    dmap = Basemap(projection='cyl',
                   llcrnrlat=map_lat[0], llcrnrlon=map_lon[0],
                   urcrnrlat=map_lat[1], urcrnrlon=map_lon[1],
                   resolution='h',ax=ax)# resolution: c,l,i,h,f.
    dmap.drawparallels(np.arange(int(map_lat[0])-1,
                                 int(map_lat[1])+1,interval_lat),
                       labels=[1,0,0,0])
    dmap.drawmeridians(np.arange(int(map_lon[0])-1,
                                 int(map_lon[1])+1,interval_lon),
                       labels=[0,0,0,1])
    #dmap.drawcoastlines()
    #dmap.fillcontinents(color='grey')
    dmap.drawmapboundary()
    #dmap.etopo()
    
class get_drifter():

    def __init__(self, drifter_id, filename=None):
        self.drifter_id = drifter_id
        self.filename = filename
        
    def getrawdrift(self, did,filename):
        
        '''
        routine to get raw drifter data from ascii files posted on the web
        '''
        url='http://nefsc.noaa.gov/drifter/'+filename
        df=pd.read_csv(url,header=None, delimiter="\s+")
        # make a datetime
        dtime=[]
        index = np.where(df[0]==int(did))[0]
        newData = df.ix[index]
        for k in newData[0].index:
            dt1=datetime(2016, newData[2][k],newData[3][k],newData[4][k],newData[5][k])
            dtime.append(dt1)
        #print dtime
        return newData[8],newData[7],dtime,newData[9] # lat,lon,time,

    def getdrift(self,did):
        
        """
        routine to get drifter data from archive based on drifter id (did)
        -assumes "import pandas as pd" has been issued above
        -get remotely-stored drifter data via ERDDAP
        -input: deployment id ("did") number where "did" is a string
        -output: time(datetime), lat (decimal degrees), lon (decimal degrees), depth (meters)
        
        note: there is another function below called "data_extracted" that does a similar thing returning a dictionary
        
        Jim Manning June 2014
        """
        url = 'http://comet.nefsc.noaa.gov:8080/erddap/tabledap/drifters.csv?time,latitude,longitude,depth&id="'+did+'"&orderBy("time")'
        df=pd.read_csv(url,skiprows=[1]) #returns a dataframe with all that requested
        #print df    
        # generate this datetime 
        for k in range(len(df)):
           df.time[k]=parse(df.time[k]) # note this "parse" routine magically converts ERDDAP time to Python datetime
        return df.latitude.values,df.longitude.values,df.time.values,df.depth.values 
        
    def get_track(self, starttime=None, days=None):
        '''
        return drifter nodes
        if starttime is given, return nodes started from starttime
        if both starttime and days are given, return nodes of the specific time period
        '''
        if self.filename:
            temp = self.getrawdrift(self.drifter_id,self.filename)
        else:
            temp = self.getdrift(self.drifter_id)
        nodes = {}
        nodes['lon'] = np.array(temp[1])
        nodes['lat'] = np.array(temp[0])
        nodes['time'] = np.array(temp[2])
        #starttime = np.array(temp[2][0])
        if not starttime:
            starttime = np.array(temp[2][0])
        if days:
            endtime = starttime + timedelta(days=days)
            i = self.__cmptime(starttime, nodes['time'])
            j = self.__cmptime(endtime, nodes['time'])
            
            if i == j :
                print 'No data in given duration.'
                raise Exception()
                #return nodes
            nodes['lon'] = nodes['lon'][i:j+1]
            nodes['lat'] = nodes['lat'][i:j+1]
            nodes['time'] = nodes['time'][i:j+1]
        else:
            i = self.__cmptime(starttime, nodes['time'])
            nodes['lon'] = nodes['lon'][i:-1]
            nodes['lat'] = nodes['lat'][i:-1]
            nodes['time'] = nodes['time'][i:-1]
        return nodes
        
    def __cmptime(self, time, times):
        '''
        return indies of specific or nearest time in times.
        '''
        tdelta = []
        #print len(times)
        for t in times:
            tdelta.append(abs((time-t).total_seconds()))
            
        index = tdelta.index(min(tdelta))
        
        return index
