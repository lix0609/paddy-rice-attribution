# -*- coding: utf-8 -*-
import os, json
import numpy as np, pandas as pd
import matplotlib as mpl; mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.linewidth':0.8,
                     'savefig.dpi':300,'figure.dpi':120})
ROOT='/sessions/vibrant-bold-franklin/mnt/水稻'
MID=ROOT+'/增温驱动吉林水田扩张与增产_1985-2020/02_中间数据'
FIG=ROOT+'/增温驱动吉林水田扩张与增产_1985-2020/03_figures'; os.makedirs(FIG,exist_ok=True)
C={'rs':'#2c7fb8','yb':'#7fcdbb','prod':'#d95f0e','yld':'#31a354','temp':'#e34a33','prec':'#3182bd'}

prov=pd.read_csv(MID+'/paddy_area_province.csv') if os.path.exists(MID+'/paddy_area_province.csv') else None
pa=pd.read_csv(MID+'/paddy_area_by_county_1985_2020.csv'); rs=pa.groupby('year')['paddy_ha'].sum().reset_index()
yb=pd.read_csv(MID+'/yield_panel_province.csv')
mig=pd.read_csv(MID+'/centroid_migration.csv')
clim=pd.read_csv(MID+'/climate_province.csv')

# ---------- Fig.2 面积/总产/单产 ----------
fig,ax=plt.subplots(1,3,figsize=(11,3.4))
ax[0].plot(rs.year,rs.paddy_ha/1e4,'o-',ms=3,color=C['rs'],label='Satellite map')
ax[0].plot(yb.year,yb.area_ha/1e4,'s--',ms=3,color=C['yb'],label='Statistical yearbook')
ax[0].set_ylabel('Paddy area (10$^4$ ha)'); ax[0].legend(frameon=False,fontsize=8); ax[0].set_title('(a) Paddy area')
ax[1].plot(yb.year,yb.prod_kg/1e9,'o-',ms=3,color=C['prod']); ax[1].set_ylabel('Production (10$^9$ kg)'); ax[1].set_title('(b) Rice production')
ax[2].plot(yb.year,yb.yield_kg_ha/1000,'o-',ms=3,color=C['yld']); ax[2].set_ylabel('Unit yield (t ha$^{-1}$)'); ax[2].set_title('(c) Unit yield')
for a in ax: a.set_xlabel('Year'); a.spines[['top','right']].set_visible(False); a.xaxis.set_major_locator(MaxNLocator(6))
plt.tight_layout(); plt.savefig(FIG+'/Fig2_area_production_yield.png',bbox_inches='tight'); plt.close()

# ---------- Fig.3 质心北移 + 高程下移 ----------
fig,ax=plt.subplots(1,2,figsize=(7.6,3.4))
ax[0].plot(mig.year,mig.cent_lat,'o-',ms=3,color=C['rs'])
ax[0].set_ylabel('Area-weighted latitude (°N)'); ax[0].set_title('(a) Northward migration')
ax2=ax[1]; ax2.plot(mig.year,mig.mean_elev,'o-',ms=3,color=C['prod'])
ax2.set_ylabel('Area-weighted elevation (m)'); ax2.set_title('(b) Shift to lower plains')
for a in ax: a.set_xlabel('Year'); a.spines[['top','right']].set_visible(False); a.xaxis.set_major_locator(MaxNLocator(6))
plt.tight_layout(); plt.savefig(FIG+'/Fig3_centroid_migration.png',bbox_inches='tight'); plt.close()

# ---------- Fig.4 气候趋势 ----------
fig,ax=plt.subplots(1,2,figsize=(7.6,3.4))
z=np.polyfit(clim.year,clim.gs_temp,1)
ax[0].plot(clim.year,clim.gs_temp,'o-',ms=3,color=C['temp'])
ax[0].plot(clim.year,np.polyval(z,clim.year),'--',color='k',lw=1,label=f'{z[0]*10:.2f} °C/decade')
ax[0].set_ylabel('Growing-season temp (°C)'); ax[0].set_title('(a) Warming'); ax[0].legend(frameon=False,fontsize=8)
mask=~clim.gs_prec.isna(); zp=np.polyfit(clim.year[mask],clim.gs_prec[mask],1)
ax[1].plot(clim.year,clim.gs_prec,'o-',ms=3,color=C['prec'])
ax[1].plot(clim.year,np.polyval(zp,clim.year),'--',color='k',lw=1,label=f'{zp[0]*10:.1f} mm/decade')
ax[1].set_ylabel('Growing-season precip (mm)'); ax[1].set_title('(b) Precipitation'); ax[1].legend(frameon=False,fontsize=8)
for a in ax: a.set_xlabel('Year'); a.spines[['top','right']].set_visible(False); a.xaxis.set_major_locator(MaxNLocator(6))
plt.tight_layout(); plt.savefig(FIG+'/Fig4_climate_trend.png',bbox_inches='tight'); plt.close()

# ---------- Fig.5 LMDI ----------
h=json.load(open(MID+'/headline_numbers.json',encoding='utf-8'))['LMDI']
fig,ax=plt.subplots(figsize=(4.2,3.6))
vals=[h['area_effect_kg']/1e9,h['yield_effect_kg']/1e9]
b=ax.bar(['Area\neffect','Yield\neffect'],vals,color=[C['rs'],C['yld']],width=0.55)
for rect,v,s in zip(b,vals,[h['area_share_%'],h['yield_share_%']]):
    ax.text(rect.get_x()+rect.get_width()/2,v+0.05,f'{v:.2f}\n({s:.0f}%)',ha='center',fontsize=9)
ax.set_ylabel('Contribution to Δproduction (10$^9$ kg)')
ax.set_title('LMDI decomposition 1985–2020'); ax.spines[['top','right']].set_visible(False)
ax.set_ylim(0,max(vals)*1.25)
plt.tight_layout(); plt.savefig(FIG+'/Fig5_LMDI.png',bbox_inches='tight'); plt.close()
print('figures saved:', os.listdir(FIG))
