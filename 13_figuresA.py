# -*- coding: utf-8 -*-
"""学术升级版图件 A: Fig2/3/4/5/6/7/S1。editable-text PDF+PNG。"""
import json,numpy as np,pandas as pd,geopandas as gpd,rasterio,warnings; warnings.filterwarnings('ignore')
from rasterio.warp import calculate_default_transform,reproject,Resampling
from rasterio.transform import array_bounds
import matplotlib as mpl; mpl.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from matplotlib.collections import LineCollection
mpl.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial','DejaVu Sans'],'svg.fonttype':'none','pdf.fonttype':42,
 'font.size':7,'axes.titlesize':8,'axes.labelsize':7,'xtick.labelsize':6.5,'ytick.labelsize':6.5,'legend.fontsize':6.5,
 'axes.linewidth':0.7,'axes.spines.top':False,'axes.spines.right':False,'legend.frameon':False,'lines.linewidth':1.1})
OI={'blue':'#0072B2','orange':'#E69F00','green':'#009E73','verm':'#D55E00','sky':'#56B4E9','yellow':'#F0E442','purple':'#CC79A7','grey':'#7A7A7A'}
MM=1/25.4; SC=89*MM; DC=183*MM
ROOT='/sessions/vibrant-bold-franklin/mnt/水稻'; MID=ROOT+'/增温驱动吉林水田扩张与增产_1985-2020/02_中间数据'; PUB=ROOT+'/增温驱动吉林水田扩张与增产_1985-2020/03_figures/pub'; SHP=ROOT+'/吉林省_区县行政边界/吉林省_区县_合并50.shp'
def save(fig,n): fig.savefig(PUB+'/'+n+'.pdf',bbox_inches='tight'); fig.savefig(PUB+'/'+n+'.png',dpi=600,bbox_inches='tight'); fig.savefig(PUB+'/'+n+'.svg',bbox_inches='tight'); plt.close(fig)
def PL(a,L): a.text(-0.17,1.06,L,transform=a.transAxes,fontsize=9,fontweight='bold',va='top')
def sen(x,y):
    from itertools import combinations; x=np.asarray(x,float);y=np.asarray(y,float);m=~np.isnan(y);x,y=x[m],y[m]
    s=np.median([(y[j]-y[i])/(x[j]-x[i]) for i,j in combinations(range(len(x)),2)]); icpt=np.median(y)-s*np.median(x); return s,icpt
def star(p): return '***' if p<0.001 else ('**' if p<0.01 else ('*' if p<0.05 else 'n.s.'))
mk=json.load(open(MID+'/mk_trends.json'))
pa=pd.read_csv(MID+'/paddy_area_by_county_1985_2020.csv'); pa['CountyCode']=pa['CountyCode'].astype(str); rs=pa.groupby('year')['paddy_ha'].sum().reset_index()
yb=pd.read_csv(MID+'/yield_panel_province.csv'); mig=pd.read_csv(MID+'/centroid_migration.csv').sort_values('year'); clim=pd.read_csv(MID+'/climate_province.csv').sort_values('year')
h=json.load(open(MID+'/headline_numbers.json')); rr=json.load(open(MID+'/regression_results.json'))

# ===== Fig2: 面积/产量/单产 + Sen趋势 + MK星 =====
fig,ax=plt.subplots(1,3,figsize=(DC,DC*0.30))
def trendpanel(a,x,y,c,ylab,ttl,mkkey=None,extra=None):
    a.plot(x,y,'o-',color=c,mfc=c,mec='white',mew=0.4,ms=3,zorder=3)
    s,ic=sen(x,y); xr=np.array([min(x),max(x)]); a.plot(xr,ic+s*xr,'--',color='0.35',lw=0.9,zorder=2)
    a.set_ylabel(ylab); a.set_title(ttl); a.set_xlabel('Year'); a.xaxis.set_major_locator(MaxNLocator(5))
    if mkkey: a.text(0.05,0.92,f'MK {star(mk[mkkey]["MK_p"])}',transform=a.transAxes,fontsize=6.3,color='0.3')
trendpanel(ax[0],rs.year,rs.paddy_ha/1e4,OI['blue'],'Paddy area (10$^4$ ha)','Paddy area','paddy_area')
ax[0].plot(yb.year,yb.area_ha/1e4,'s',color=OI['sky'],mec='white',mew=0.3,ms=3,label='Yearbook',zorder=3)
ax[0].plot([],[],'o-',color=OI['blue'],label='Satellite'); ax[0].legend(loc='lower right')
yy=yb.dropna(subset=['prod_kg']); trendpanel(ax[1],yy.year,yy.prod_kg/1e9,OI['verm'],'Production (10$^9$ kg)','Rice production')
yz=yb.dropna(subset=['yield_kg_ha']); trendpanel(ax[2],yz.year,yz.yield_kg_ha/1000,OI['green'],'Unit yield (t ha$^{-1}$)','Unit yield')
for a,L in zip(ax,'abc'): PL(a,L)
fig.tight_layout(); save(fig,'Fig2_area_production_yield')

# ===== Fig3: 质心迁移轨迹图(hero) + 纬度/高程双序列 =====
fig=plt.figure(figsize=(DC*0.72,DC*0.32)); gs=fig.add_gridspec(1,2,width_ratios=[1,1.15])
a0=fig.add_subplot(gs[0])
pts=np.array([mig.cent_lon,mig.cent_lat]).T.reshape(-1,1,2); segs=np.concatenate([pts[:-1],pts[1:]],axis=1)
lc=LineCollection(segs,cmap='viridis',linewidth=2); lc.set_array(mig.year.values[:-1]); a0.add_collection(lc)
sczz=a0.scatter(mig.cent_lon,mig.cent_lat,c=mig.year,cmap='viridis',s=14,zorder=3,edgecolor='white',lw=0.3)
a0.annotate('1985',(mig.cent_lon.iloc[0],mig.cent_lat.iloc[0]),fontsize=6.3,ha='right'); a0.annotate('2020',(mig.cent_lon.iloc[-1],mig.cent_lat.iloc[-1]),fontsize=6.3,ha='left')
a0.set_xlabel('Centroid longitude (°E)'); a0.set_ylabel('Centroid latitude (°N)'); a0.set_title('Centroid trajectory')
cb=fig.colorbar(sczz,ax=a0,shrink=0.8,pad=0.02); cb.set_label('Year',fontsize=6.3); PL(a0,'a')
a1=fig.add_subplot(gs[1]); a1.plot(mig.year,mig.cent_lat,'o-',color=OI['blue'],ms=3,mec='white',mew=0.3,label='Latitude')
s,ic=sen(mig.year,mig.cent_lat); a1.plot([mig.year.min(),mig.year.max()],[ic+s*mig.year.min(),ic+s*mig.year.max()],'--',color='0.35',lw=0.9)
a1.set_ylabel('Latitude (°N)',color=OI['blue']); a1.tick_params(axis='y',colors=OI['blue']); a1.set_xlabel('Year')
a1.text(0.04,0.9,f'+0.74°, ≈82 km {star(mk["cent_lat"]["MK_p"])}',transform=a1.transAxes,fontsize=6.3,color=OI['blue'])
a2=a1.twinx(); a2.plot(mig.year,mig.mean_elev,'s-',color=OI['orange'],ms=3,mec='white',mew=0.3)
a2.set_ylabel('Elevation (m)',color=OI['orange']); a2.tick_params(axis='y',colors=OI['orange']); a2.spines['top'].set_visible(False)
a2.text(0.04,0.10,f'−99 m {star(mk["mean_elev"]["MK_p"])}',transform=a2.transAxes,fontsize=6.3,color=OI['orange'])
a1.set_title('Latitude and elevation'); PL(a1,'b')
save(fig,'Fig3_centroid_migration')

# ===== Fig4: 生长季气候距平条 + Sen趋势 =====
fig,ax=plt.subplots(1,2,figsize=(DC*0.66,DC*0.30))
tm=clim.gs_temp.mean(); da=clim.gs_temp-tm; cols=[OI['verm'] if v>=0 else OI['blue'] for v in da]
ax[0].bar(clim.year,da,color=cols,width=0.8,edgecolor='none')
s,ic=sen(clim.year,clim.gs_temp); ax[0].plot(clim.year,(ic+s*clim.year)-tm,'-',color='0.2',lw=1)
ax[0].axhline(0,color='k',lw=0.5); ax[0].set_ylabel('GS temp. anomaly (°C)'); ax[0].set_title('Growing-season warming')
ax[0].text(0.04,0.9,f'+0.35 °C dec$^{{-1}}$ {star(mk["gs_temp"]["MK_p"])}',transform=ax[0].transAxes,fontsize=6.3)
pm=clim.gs_prec.mean(); dp=clim.gs_prec-pm; colp=[OI['blue'] if v>=0 else OI['orange'] for v in dp]
ax[1].bar(clim.year,dp,color=colp,width=0.8); ax[1].axhline(0,color='k',lw=0.5)
s,ic=sen(clim.year,clim.gs_prec); ax[1].plot(clim.year,(ic+s*clim.year)-pm,'-',color='0.2',lw=1)
ax[1].set_ylabel('GS precip. anomaly (mm)'); ax[1].set_title('Growing-season precipitation')
ax[1].text(0.04,0.9,f'−2.1 mm yr$^{{-1}}$ {star(mk["gs_prec"]["MK_p"])}',transform=ax[1].transAxes,fontsize=6.3)
for a,L in zip(ax,'ab'): a.set_xlabel('Year'); a.xaxis.set_major_locator(MaxNLocator(5)); PL(a,L)
fig.tight_layout(); save(fig,'Fig4_climate_trend')

# ===== Fig5: LMDI 瀑布图 =====
L=h['LMDI']; P0=h['YB_prod_1985']/1e9; Ae=L['area_effect_kg']/1e9; Ye=L['yield_effect_kg']/1e9; P1=h['YB_prod_2020']/1e9
fig,ax=plt.subplots(figsize=(SC*1.15,SC*0.92))
labels=['1985\nproduction','Area\neffect','Yield\neffect','2020\nproduction']
bottoms=[0,P0,P0+Ae,0]; heights=[P0,Ae,Ye,P1]; colors=[OI['grey'],OI['blue'],OI['green'],OI['grey']]
for i,(b,hh,c) in enumerate(zip(bottoms,heights,colors)):
    ax.bar(i,hh,bottom=b,color=c,width=0.62,edgecolor='white')
# connectors
for i,(cum) in enumerate([P0,P0+Ae,P1]):
    ax.plot([i+0.31,i+1-0.31],[cum,cum],color='0.5',lw=0.7,ls='--')
for i,(b,hh) in enumerate(zip(bottoms,heights)):
    lab=f'{hh:.2f}' + (f'\n({L["area_share_%"]:.0f}%)' if i==1 else (f'\n({L["yield_share_%"]:.0f}%)' if i==2 else ''))
    ax.text(i,b+hh+0.12,lab,ha='center',fontsize=6.5)
ax.set_xticks(range(4)); ax.set_xticklabels(labels); ax.set_ylabel('Rice production (10$^9$ kg)')
ax.set_title('LMDI decomposition of production growth'); ax.set_ylim(0,P1*1.16)
save(fig,'Fig5_LMDI')

# ===== Fig7: (a)扩张倍数地图 (b)棒棒糖:分带扩张 + 增温标注 =====
g=gpd.read_file(SHP).to_crs(4326); g['CountyCode']=g['CountyCode'].astype(str)
cc=pd.read_csv(MID+'/county_change_summary.csv'); cc['CountyCode']=cc['CountyCode'].astype(str)
bands=pd.DataFrame(rr['latitude_bands']).set_index('band').loc[['South','Central','North']]
m=g.merge(cc[['CountyCode','expansion_x']],on='CountyCode',how='left'); m['exp_cl']=m['expansion_x'].clip(upper=8)
fig,ax=plt.subplots(1,2,figsize=(DC*0.72,DC*0.32))
m.plot(column='exp_cl',ax=ax[0],cmap='YlOrRd',legend=True,edgecolor='0.5',lw=0.25,legend_kwds={'label':'Expansion factor (capped 8)','shrink':0.68})
ax[0].set_title('County expansion factor'); ax[0].axis('off'); PL(ax[0],'a')
yb2=np.arange(3); ex=bands['expansion_x'].values; wm=bands['warming_C_dec'].values
ax[1].hlines(yb2,1,ex,color=OI['grey'],lw=1.4,zorder=1)
sc=ax[1].scatter(ex,yb2,c=wm,cmap='OrRd',s=90,zorder=3,edgecolor='k',lw=0.5,vmin=0.32,vmax=0.37)
for i,(e,w) in enumerate(zip(ex,wm)): ax[1].text(e+0.15,i,f'{e:.1f}×',va='center',fontsize=7)
ax[1].set_yticks(yb2); ax[1].set_yticklabels(bands.index); ax[1].set_xlabel('Expansion factor (2020/1985)')
ax[1].set_xlim(0.8,6.2); ax[1].set_title('Expansion by latitude band'); ax[1].axvline(1,color='0.7',lw=0.6,ls=':')
cb=fig.colorbar(sc,ax=ax[1],shrink=0.7,pad=0.02); cb.set_label('Warming (°C dec$^{-1}$)',fontsize=6.3); PL(ax[1],'b')
fig.tight_layout(); save(fig,'Fig7_expansion_vs_warming')
print('Fig2,3,4,5,7 done')

# ===== Fig6: 三期地图 + 净变化(第4面板) =====
fig,axes=plt.subplots(1,4,figsize=(DC,DC*0.30))
vmax=pa[pa.year.isin([1985,2000,2020])]['paddy_ha'].quantile(0.98)/1e4
for a,yr,Lc in zip(axes[:3],[1985,2000,2020],'abc'):
    mm=g.merge(pa[pa.year==yr][['CountyCode','paddy_ha']],on='CountyCode',how='left'); mm['pk']=mm['paddy_ha']/1e4
    mm.plot(column='pk',ax=a,cmap='YlGnBu',vmin=0,vmax=vmax,edgecolor='0.6',lw=0.2); a.set_title(str(yr)); a.axis('off'); PL(a,Lc)
d85=pa[pa.year==1985][['CountyCode','paddy_ha']].rename(columns={'paddy_ha':'p85'}); d20=pa[pa.year==2020][['CountyCode','paddy_ha']].rename(columns={'paddy_ha':'p20'})
ch=g.merge(d85,on='CountyCode').merge(d20,on='CountyCode'); ch['dpk']=(ch['p20']-ch['p85'])/1e4
ch.plot(column='dpk',ax=axes[3],cmap='RdYlGn',edgecolor='0.6',lw=0.2,legend=True,legend_kwds={'label':'Δ paddy (10$^4$ ha)','shrink':0.6}); axes[3].set_title('Net change'); axes[3].axis('off'); PL(axes[3],'d')
sm=mpl.cm.ScalarMappable(cmap='YlGnBu',norm=mpl.colors.Normalize(0,vmax)); cb=fig.colorbar(sm,ax=axes[:3],shrink=0.5,label='Paddy area (10$^4$ ha)',pad=0.01)
save(fig,'Fig6_paddy_expansion_maps')
print('Fig6 done')
