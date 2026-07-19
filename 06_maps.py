# -*- coding: utf-8 -*-
import os, warnings
import numpy as np, pandas as pd, geopandas as gpd
import matplotlib as mpl; mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LightSource
import rasterio
from rasterio.enums import Resampling
warnings.filterwarnings('ignore')
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'savefig.dpi':300})
ROOT='/sessions/vibrant-bold-franklin/mnt/水稻'
MID=ROOT+'/增温驱动吉林水田扩张与增产_1985-2020/02_中间数据'
FIG=ROOT+'/增温驱动吉林水田扩张与增产_1985-2020/03_figures'
SHP=ROOT+'/吉林省_区县行政边界/吉林省_区县_合并50.shp'
DEM=ROOT+'/3.空间数据预处理/DEM/dem_jl.tif'

g=gpd.read_file(SHP).to_crs(4326); g['CountyCode']=g['CountyCode'].astype(str)
pa=pd.read_csv(MID+'/paddy_area_by_county_1985_2020.csv'); pa['CountyCode']=pa['CountyCode'].astype(str)
cc=pd.read_csv(MID+'/county_change_summary.csv'); cc['CountyCode']=cc['CountyCode'].astype(str)

# DEM 降采样
with rasterio.open(DEM) as d:
    sc=max(1,int(max(d.width,d.height)/1200))
    dem=d.read(1,out_shape=(d.height//sc,d.width//sc),resampling=Resampling.average).astype('float32')
    if d.nodata is not None: dem[dem==d.nodata]=np.nan
    dem[dem<-100]=np.nan
    b=d.bounds; import rasterio.warp as rw
    # DEM CRS may not be 4326; get extent in its crs then transform corners
    extent=[b.left,b.right,b.bottom,b.top]; demcrs=d.crs

# ---------- Fig.1 研究区 ----------
fig,ax=plt.subplots(figsize=(7,5))
ls=LightSource(azdeg=315,altdeg=45)
if str(demcrs).upper() not in ('EPSG:4326',):
    # reproject dem extent corners to 4326 for placing
    import rasterio.warp as rw
    xs,ys=rw.transform(demcrs,'EPSG:4326',[extent[0],extent[1]],[extent[2],extent[3]])
    ext4326=[xs[0],xs[1],ys[0],ys[1]]
else:
    ext4326=extent
im=ax.imshow(dem,extent=ext4326,cmap='terrain',alpha=0.85,origin='upper',aspect='auto')
g.boundary.plot(ax=ax,color='k',lw=0.4)
# 2020 水田密度叠加(choropleth by county, semi-transparent)
p20=g.merge(pa[pa.year==2020][['CountyCode','paddy_ha']],on='CountyCode',how='left')
p20['dens']=p20['paddy_ha']/(p20.to_crs(3395).area/1e6)  # ha per km2
plt.colorbar(im,ax=ax,shrink=0.7,label='Elevation (m)')
ax.set_xlabel('Longitude (°E)'); ax.set_ylabel('Latitude (°N)')
ax.set_title('Study area: Jilin Province (49 counties) with terrain')
plt.tight_layout(); plt.savefig(FIG+'/Fig1_study_area.png',bbox_inches='tight'); plt.close()

# ---------- Fig6 三期水田面积 choropleth ----------
fig,axes=plt.subplots(1,3,figsize=(13,3.8))
vmax=pa[pa.year.isin([1985,2000,2020])]['paddy_ha'].quantile(0.98)/1e4
for ax,yr in zip(axes,[1985,2000,2020]):
    m=g.merge(pa[pa.year==yr][['CountyCode','paddy_ha']],on='CountyCode',how='left')
    m['pk']=m['paddy_ha']/1e4
    m.plot(column='pk',ax=ax,cmap='YlGnBu',vmin=0,vmax=vmax,legend=False,edgecolor='grey',lw=0.3)
    ax.set_title(f'{yr}'); ax.set_xticks([]); ax.set_yticks([])
sm=mpl.cm.ScalarMappable(cmap='YlGnBu',norm=mpl.colors.Normalize(0,vmax))
fig.colorbar(sm,ax=axes,shrink=0.7,label='Paddy area (10$^4$ ha)')
fig.suptitle('Spatiotemporal expansion of paddy rice, Jilin 1985–2020',y=1.02)
plt.savefig(FIG+'/Fig6_paddy_expansion_maps.png',bbox_inches='tight'); plt.close()

# ---------- Fig7 扩张倍数地图 + 增温散点 ----------
fig,ax=plt.subplots(1,2,figsize=(11,4))
m=g.merge(cc[['CountyCode','expansion_x','warming_C_dec','lat']],on='CountyCode',how='left')
m['exp_cl']=m['expansion_x'].clip(upper=8)
m.plot(column='exp_cl',ax=ax[0],cmap='OrRd',legend=True,edgecolor='grey',lw=0.3,
       legend_kwds={'label':'Expansion factor (2020/1985)','shrink':0.7})
ax[0].set_title('(a) County expansion factor'); ax[0].set_xticks([]); ax[0].set_yticks([])
s=cc.dropna(subset=['warming_C_dec','expansion_x'])
sc=ax[1].scatter(s.warming_C_dec,s.expansion_x,c=s.lat,cmap='viridis',s=28,edgecolor='k',lw=0.3)
ax[1].set_xlabel('Warming (°C decade$^{-1}$)'); ax[1].set_ylabel('Expansion factor')
ax[1].set_title('(b) Warming vs expansion'); ax[1].spines[['top','right']].set_visible(False)
plt.colorbar(sc,ax=ax[1],label='Latitude (°N)',shrink=0.8)
plt.tight_layout(); plt.savefig(FIG+'/Fig7_expansion_vs_warming.png',bbox_inches='tight'); plt.close()
print('maps saved:',[f for f in os.listdir(FIG) if f.startswith(('Fig1','Fig6','Fig7'))])
