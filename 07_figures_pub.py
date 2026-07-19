# -*- coding: utf-8 -*-
"""Nature-style publication figures: Okabe-Ito palette, unified typography, vector PDF + 600 dpi PNG."""
import os, json, warnings
import numpy as np, pandas as pd, geopandas as gpd, rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio.transform import array_bounds
import matplotlib as mpl; mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
warnings.filterwarnings('ignore')
# ---- Nature-ish rcParams ----
plt.rcParams.update({
 'font.family':'sans-serif','font.sans-serif':['DejaVu Sans','Arial','Helvetica'],
 'font.size':7,'axes.titlesize':8,'axes.labelsize':7,'xtick.labelsize':6.5,'ytick.labelsize':6.5,
 'legend.fontsize':6.5,'axes.linewidth':0.6,'xtick.major.width':0.6,'ytick.major.width':0.6,
 'xtick.major.size':2.5,'ytick.major.size':2.5,'lines.linewidth':1.0,'lines.markersize':3,
 'savefig.dpi':600,'figure.dpi':150,'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none'})
# Okabe-Ito colorblind-safe
OI={'blue':'#0072B2','orange':'#E69F00','green':'#009E73','vermillion':'#D55E00',
    'skyblue':'#56B4E9','yellow':'#F0E442','purple':'#CC79A7','black':'#000000'}
MM=1/25.4; SC=89*MM; DC=183*MM  # single/double column widths (inch)
ROOT='/sessions/vibrant-bold-franklin/mnt/水稻'; MID=ROOT+'/增温驱动吉林水田扩张与增产_1985-2020/02_中间数据'
FIG=ROOT+'/增温驱动吉林水田扩张与增产_1985-2020/03_figures'; PUB=FIG+'/pub'; os.makedirs(PUB,exist_ok=True)
def save(fig,name): 
    fig.savefig(PUB+'/'+name+'.pdf',bbox_inches='tight'); fig.savefig(PUB+'/'+name+'.png',bbox_inches='tight',dpi=600); plt.close(fig)
def despine(ax):
    for s in ['top','right']: ax.spines[s].set_visible(False)
def pl(ax,x,y,c,**k): ax.plot(x,y,'o-',color=c,mfc=c,mec='white',mew=0.3,**k)
def panel(ax,letter): ax.text(-0.16,1.05,letter,transform=ax.transAxes,fontsize=9,fontweight='bold',va='top')

pa=pd.read_csv(MID+'/paddy_area_by_county_1985_2020.csv'); pa['CountyCode']=pa['CountyCode'].astype(str); rs=pa.groupby('year')['paddy_ha'].sum().reset_index()
yb=pd.read_csv(MID+'/yield_panel_province.csv'); mig=pd.read_csv(MID+'/centroid_migration.csv'); clim=pd.read_csv(MID+'/climate_province.csv')
mk=json.load(open(MID+'/mk_trends.json')); h=json.load(open(MID+'/headline_numbers.json')); rr=json.load(open(MID+'/regression_results.json'))
def pstar(p): return '***' if p<0.001 else ('**' if p<0.01 else ('*' if p<0.05 else 'n.s.'))

# ===== Fig2 =====
fig,ax=plt.subplots(1,3,figsize=(DC,DC*0.30))
pl(ax[0],rs.year,rs.paddy_ha/1e4,OI['blue'],label='Satellite map')
ax[0].plot(yb.year,yb.area_ha/1e4,'s--',color=OI['skyblue'],mfc=OI['skyblue'],mec='white',mew=0.3,label='Yearbook')
ax[0].set_ylabel('Paddy area (10$^4$ ha)'); ax[0].legend(frameon=False); ax[0].set_title('Paddy area')
ax[0].text(0.5,0.06,f"MK {pstar(mk['paddy_area']['MK_p'])}",transform=ax[0].transAxes,fontsize=6,color='0.3')
pl(ax[1],yb.year,yb.prod_kg/1e9,OI['vermillion']); ax[1].set_ylabel('Production (10$^9$ kg)'); ax[1].set_title('Rice production')
pl(ax[2],yb.year,yb.yield_kg_ha/1000,OI['green']); ax[2].set_ylabel('Unit yield (t ha$^{-1}$)'); ax[2].set_title('Unit yield')
for a,L in zip(ax,'abc'): a.set_xlabel('Year'); despine(a); a.xaxis.set_major_locator(MaxNLocator(5)); panel(a,L)
fig.tight_layout(); save(fig,'Fig2_area_production_yield')

# ===== Fig3 =====
fig,ax=plt.subplots(1,2,figsize=(DC*0.66,DC*0.30))
pl(ax[0],mig.year,mig.cent_lat,OI['blue']); ax[0].set_ylabel('Area-weighted latitude (°N)'); ax[0].set_title('Northward migration')
ax[0].text(0.05,0.9,f"+0.74° (≈82 km)  {pstar(mk['cent_lat']['MK_p'])}",transform=ax[0].transAxes,fontsize=6.5,color='0.2')
pl(ax[1],mig.year,mig.mean_elev,OI['orange']); ax[1].set_ylabel('Area-weighted elevation (m)'); ax[1].set_title('Shift to lower plains')
ax[1].text(0.05,0.12,f"−99 m  {pstar(mk['mean_elev']['MK_p'])}",transform=ax[1].transAxes,fontsize=6.5,color='0.2')
for a,L in zip(ax,'ab'): a.set_xlabel('Year'); despine(a); a.xaxis.set_major_locator(MaxNLocator(5)); panel(a,L)
fig.tight_layout(); save(fig,'Fig3_centroid_migration')

# ===== Fig4 =====
def sen(x,y):
    from itertools import combinations; x=np.asarray(x,float);y=np.asarray(y,float);m=~np.isnan(y);x,y=x[m],y[m]
    s=np.median([(y[j]-y[i])/(x[j]-x[i]) for i,j in combinations(range(len(x)),2)]); return s,x,y
fig,ax=plt.subplots(1,2,figsize=(DC*0.66,DC*0.30))
st,xt,yt=sen(clim.year,clim.gs_temp); xr=np.array([clim.year.min(),clim.year.max()])
pl(ax[0],clim.year,clim.gs_temp,OI['vermillion'])
ax[0].plot(xr,(np.median(yt)-st*np.median(xt))+st*xr,'--',color='0.2',lw=0.9)
ax[0].set_ylabel('Growing-season T (°C)'); ax[0].set_title('Warming')
ax[0].text(0.05,0.9,f"+0.35 °C dec$^{{-1}}$  {pstar(mk['gs_temp']['MK_p'])}",transform=ax[0].transAxes,fontsize=6.5,color='0.2')
sp,xp,yp=sen(clim.year,clim.gs_prec); pl(ax[1],clim.year,clim.gs_prec,OI['blue'])
ax[1].plot(xr,(np.median(yp)-sp*np.median(xp))+sp*xr,'--',color='0.2',lw=0.9)
ax[1].set_ylabel('Growing-season P (mm)'); ax[1].set_title('Precipitation')
ax[1].text(0.05,0.9,f"−2.1 mm yr$^{{-1}}$  {pstar(mk['gs_prec']['MK_p'])}",transform=ax[1].transAxes,fontsize=6.5,color='0.2')
for a,L in zip(ax,'ab'): a.set_xlabel('Year'); despine(a); a.xaxis.set_major_locator(MaxNLocator(5)); panel(a,L)
fig.tight_layout(); save(fig,'Fig4_climate_trend')

# ===== Fig5 =====
L=h['LMDI']; fig,ax=plt.subplots(figsize=(SC,SC*0.85))
vals=[L['area_effect_kg']/1e9,L['yield_effect_kg']/1e9]
bars=ax.bar(['Area\neffect','Yield\neffect'],vals,color=[OI['blue'],OI['green']],width=0.6,edgecolor='white')
for r,v,s in zip(bars,vals,[L['area_share_%'],L['yield_share_%']]):
    ax.text(r.get_x()+r.get_width()/2,v+0.05,f'{v:.2f}\n({s:.0f}%)',ha='center',fontsize=6.5)
ax.set_ylabel('Contribution to Δproduction (10$^9$ kg)'); ax.set_ylim(0,max(vals)*1.25); despine(ax)
ax.set_title('LMDI decomposition, 1985–2020')
fig.tight_layout(); save(fig,'Fig5_LMDI')

# ===== Fig7 (map + bands) =====
g=gpd.read_file(ROOT+'/吉林省_区县行政边界/吉林省_区县_合并50.shp').to_crs(4326); g['CountyCode']=g['CountyCode'].astype(str)
cc=pd.read_csv(MID+'/county_change_summary.csv'); cc['CountyCode']=cc['CountyCode'].astype(str)
bands=pd.DataFrame(rr['latitude_bands']).set_index('band').loc[['South','Central','North']]
m=g.merge(cc[['CountyCode','expansion_x']],on='CountyCode',how='left'); m['exp_cl']=m['expansion_x'].clip(upper=8)
fig,ax=plt.subplots(1,2,figsize=(DC*0.72,DC*0.30))
m.plot(column='exp_cl',ax=ax[0],cmap='YlOrRd',legend=True,edgecolor='0.5',lw=0.25,
       legend_kwds={'label':'Expansion factor (capped 8)','shrink':0.7})
ax[0].set_title('County expansion factor'); ax[0].axis('off'); panel(ax[0],'a')
x=np.arange(3)
ax[1].bar(x-0.2,bands['expansion_x'],0.4,color=OI['vermillion'],edgecolor='white',label='Expansion')
ax[1].set_ylabel('Expansion factor',color=OI['vermillion']); ax[1].tick_params(axis='y',colors=OI['vermillion'])
ax[1].set_xticks(x); ax[1].set_xticklabels(bands.index); ax[1].set_title('By latitude band')
for xi,v in zip(x-0.2,bands['expansion_x']): ax[1].text(xi,v+0.1,f'{v:.1f}×',ha='center',fontsize=6,color=OI['vermillion'])
ax2=ax[1].twinx(); ax2.bar(x+0.2,bands['warming_C_dec'],0.4,color=OI['blue'],edgecolor='white')
ax2.set_ylabel('Warming (°C dec$^{-1}$)',color=OI['blue']); ax2.tick_params(axis='y',colors=OI['blue']); ax2.set_ylim(0,0.45)
for xi,v in zip(x+0.2,bands['warming_C_dec']): ax2.text(xi,v+0.006,f'{v:.2f}',ha='center',fontsize=6,color=OI['blue'])
for s in ['top']: ax[1].spines[s].set_visible(False); ax2.spines[s].set_visible(False)
panel(ax[1],'b'); fig.tight_layout(); save(fig,'Fig7_expansion_vs_warming')

# ===== Fig6 maps =====
fig,axes=plt.subplots(1,3,figsize=(DC,DC*0.32))
vmax=pa[pa.year.isin([1985,2000,2020])]['paddy_ha'].quantile(0.98)/1e4
for ax,yr,L in zip(axes,[1985,2000,2020],'abc'):
    mm=g.merge(pa[pa.year==yr][['CountyCode','paddy_ha']],on='CountyCode',how='left'); mm['pk']=mm['paddy_ha']/1e4
    mm.plot(column='pk',ax=ax,cmap='YlGnBu',vmin=0,vmax=vmax,edgecolor='0.6',lw=0.2)
    ax.set_title(str(yr)); ax.axis('off'); panel(ax,L)
sm=mpl.cm.ScalarMappable(cmap='YlGnBu',norm=mpl.colors.Normalize(0,vmax))
cb=fig.colorbar(sm,ax=axes,shrink=0.6,label='Paddy area (10$^4$ ha)',pad=0.01)
save(fig,'Fig6_paddy_expansion_maps')

# ===== Fig1 study area =====
DEM=ROOT+'/3.空间数据预处理/DEM/dem_jl.tif'
with rasterio.open(DEM) as src:
    t,w,hh=calculate_default_transform(src.crs,'EPSG:4326',src.width,src.height,*src.bounds)
    scl=max(1,int(max(w,hh)/1400)); w//=scl; hh//=scl
    t,w,hh=calculate_default_transform(src.crs,'EPSG:4326',src.width,src.height,*src.bounds,dst_width=w,dst_height=hh)
    dem=np.full((hh,w),np.nan,'float32')
    reproject(rasterio.band(src,1),dem,src_transform=src.transform,src_crs=src.crs,dst_transform=t,dst_crs='EPSG:4326',resampling=Resampling.average,dst_nodata=np.nan)
dem[dem<-100]=np.nan; bb=array_bounds(hh,w,t); ext=[bb[0],bb[2],bb[1],bb[3]]
fig,ax=plt.subplots(figsize=(SC*1.5,SC*1.5*0.78))
im=ax.imshow(dem,extent=ext,cmap='terrain',origin='upper',aspect='auto',vmin=0,vmax=1400)
g.boundary.plot(ax=ax,color='0.1',lw=0.4)
ax.set_xlim(ext[0],ext[1]); ax.set_ylim(ext[2],ext[3])
ax.set_xlabel('Longitude (°E)'); ax.set_ylabel('Latitude (°N)'); ax.set_title('Study area: Jilin Province, Northeast China')
plt.colorbar(im,ax=ax,shrink=0.75,label='Elevation (m)',pad=0.02)
save(fig,'Fig1_study_area')
print('PUB figures:',sorted(os.listdir(PUB)))
