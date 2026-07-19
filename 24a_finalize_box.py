# -*- coding: utf-8 -*-
"""补全框线(top/right)重绘: Fig1,Fig2,Fig3,S2气候,S4物候,S5一致性。box默认开。"""
import json,numpy as np,pandas as pd,geopandas as gpd,rasterio,warnings; warnings.filterwarnings('ignore')
from rasterio.warp import calculate_default_transform,reproject,Resampling
from rasterio.transform import array_bounds
from matplotlib.collections import LineCollection
import matplotlib as mpl; mpl.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import pymannkendall as mk
mpl.rcParams.update({'font.family':'serif','font.serif':['Times New Roman','Liberation Serif','Nimbus Roman','DejaVu Serif'],'font.weight':'bold','axes.labelweight':'bold','axes.titleweight':'bold','svg.fonttype':'none','pdf.fonttype':42,
 'font.size':7,'axes.titlesize':7.8,'axes.labelsize':7,'xtick.labelsize':6.3,'ytick.labelsize':6.3,'legend.fontsize':6.3,
 'axes.linewidth':0.7,'legend.frameon':False})   # 不再关闭top/right → 全框
OI={'blue':'#0072B2','green':'#009E73','orange':'#E69F00','verm':'#D55E00','sky':'#56B4E9','grey':'#7A7A7A'}; MM=1/25.4; SC=89*MM; DC=183*MM
ROOT='/sessions/vibrant-bold-franklin/mnt/水稻'; PROJ=ROOT+'/增温驱动吉林水田扩张与增产_1985-2020'; MID=PROJ+'/02_中间数据'; PH=PROJ+'/06_phenology'; PUB=PROJ+'/03_figures/pub'; SHP=ROOT+'/吉林省_区县行政边界/吉林省_区县_合并50.shp'; DEM=ROOT+'/3.空间数据预处理/DEM/dem_jl.tif'
def sv(fig,n): fig.savefig(PUB+'/'+n+'.pdf',bbox_inches='tight'); fig.savefig(PUB+'/'+n+'.png',dpi=600,bbox_inches='tight'); fig.savefig(PUB+'/'+n+'.svg',bbox_inches='tight'); plt.close(fig)
def plab(ax,L): ax.annotate(L,xy=(0,1),xycoords='axes fraction',xytext=(-32,5),textcoords='offset points',fontsize=9,fontweight='bold',va='bottom')
def sen(x,y):
    from itertools import combinations;x=np.asarray(x,float);y=np.asarray(y,float);m=~np.isnan(y);x,y=x[m],y[m]
    s=np.median([(y[j]-y[i])/(x[j]-x[i]) for i,j in combinations(range(len(x)),2)]);return s,np.median(y)-s*np.median(x)
def star(p): return '***' if p<0.001 else ('**' if p<0.01 else ('*' if p<0.05 else 'n.s.'))
mk_=json.load(open(MID+'/mk_trends.json')); h=json.load(open(MID+'/headline_numbers.json'))
g=gpd.read_file(SHP).to_crs(4326); g['CountyCode']=g['CountyCode'].astype(str)
pa=pd.read_csv(MID+'/paddy_area_by_county_1985_2020.csv'); pa['CountyCode']=pa['CountyCode'].astype(str); rs=pa.groupby('year')['paddy_ha'].sum().reset_index()
yb=pd.read_csv(MID+'/yield_panel_province.csv'); mig=pd.read_csv(MID+'/centroid_migration.csv').sort_values('year'); clim=pd.read_csv(MID+'/climate_province.csv').sort_values('year')

# ===== Fig1 study area (地图有经纬轴 → 加框) =====
with rasterio.open(DEM) as src:
    t,w,hh=calculate_default_transform(src.crs,'EPSG:4326',src.width,src.height,*src.bounds); scl=max(1,int(max(w,hh)/1400)); w//=scl; hh//=scl
    t,w,hh=calculate_default_transform(src.crs,'EPSG:4326',src.width,src.height,*src.bounds,dst_width=w,dst_height=hh)
    dem=np.full((hh,w),np.nan,'float32'); reproject(rasterio.band(src,1),dem,src_transform=src.transform,src_crs=src.crs,dst_transform=t,dst_crs='EPSG:4326',resampling=Resampling.average,dst_nodata=np.nan)
dem[dem<-100]=np.nan; bb=array_bounds(hh,w,t); ext=[bb[0],bb[2],bb[1],bb[3]]
fig,ax=plt.subplots(figsize=(SC*1.5,SC*1.5*0.78))
im=ax.imshow(dem,extent=ext,cmap='terrain',origin='upper',aspect='auto',vmin=0,vmax=1400); g.boundary.plot(ax=ax,color='0.1',lw=0.4)
ax.set_xlim(ext[0],ext[1]); ax.set_ylim(ext[2],ext[3]); ax.set_xlabel('Longitude (°E)'); ax.set_ylabel('Latitude (°N)'); ax.set_title('Study area: Jilin Province, Northeast China')
plt.colorbar(im,ax=ax,shrink=0.75,label='Elevation (m)',pad=0.02); sv(fig,'Fig1_study_area'); print('Fig1 ok')

# ===== Fig2 growth+LMDI =====
fig,ax=plt.subplots(2,2,figsize=(DC*0.78,DC*0.60),constrained_layout=True)
def tp(a,x,y,c,ylab,ttl,st=None):
    a.plot(x,y,'o-',color=c,mfc=c,mec='white',mew=0.4,ms=3,zorder=3); s,ic=sen(x,y); xr=np.array([min(x),max(x)]); a.plot(xr,ic+s*xr,'--',color='0.4',lw=0.9)
    a.set_ylabel(ylab); a.set_title(ttl); a.set_xlabel('Year'); a.xaxis.set_major_locator(MaxNLocator(5))
    if st: a.text(0.05,0.9,st,transform=a.transAxes,fontsize=6.2,va='top',color='0.3')
tp(ax[0,0],rs.year,rs.paddy_ha/1e4,OI['blue'],'Paddy area (10$^4$ ha)','Paddy area','MK ***')
ax[0,0].plot(yb.year,yb.area_ha/1e4,'s',color=OI['sky'],mec='white',mew=0.3,ms=3,zorder=3)
ax[0,0].plot([],[],'o-',color=OI['blue'],label='Satellite'); ax[0,0].plot([],[],'s',color=OI['sky'],label='Yearbook'); ax[0,0].legend(loc='lower right'); plab(ax[0,0],'a')
yy=yb.dropna(subset=['prod_kg']); tp(ax[0,1],yy.year,yy.prod_kg/1e9,OI['verm'],'Production (10$^9$ kg)','Rice production'); plab(ax[0,1],'b')
yz=yb.dropna(subset=['yield_kg_ha']); tp(ax[1,0],yz.year,yz.yield_kg_ha/1000,OI['green'],'Unit yield (t ha$^{-1}$)','Unit yield'); plab(ax[1,0],'c')
L=h['LMDI']; P0=h['YB_prod_1985']/1e9; Ae=L['area_effect_kg']/1e9; Ye=L['yield_effect_kg']/1e9; P1=h['YB_prod_2020']/1e9
axd=ax[1,1]; bottoms=[0,P0,P0+Ae,0]; heights=[P0,Ae,Ye,P1]; cols=[OI['grey'],OI['blue'],OI['green'],OI['grey']]
for i,(b,hgt,c) in enumerate(zip(bottoms,heights,cols)): axd.bar(i,hgt,bottom=b,color=c,width=0.62,edgecolor='white')
for i,cum in enumerate([P0,P0+Ae]): axd.plot([i+0.31,i+1-0.31],[cum,cum],color='0.5',lw=0.7,ls='--')
for i,(b,hgt) in enumerate(zip(bottoms,heights)):
    lab=f'{hgt:.1f}'+(f'\n{L["area_share_%"]:.0f}%' if i==1 else (f'\n{L["yield_share_%"]:.0f}%' if i==2 else ''))
    axd.text(i,b+hgt+0.1,lab,ha='center',fontsize=6.2)
axd.set_xticks(range(4)); axd.set_xticklabels(['1985','Area','Yield','2020']); axd.set_ylabel('Production (10$^9$ kg)'); axd.set_title('LMDI decomposition'); axd.set_ylim(0,P1*1.18); plab(axd,'d')
sv(fig,'Fig2_growth_lmdi'); print('Fig2 ok')

# ===== Fig3 migration =====
fig,ax=plt.subplots(1,2,figsize=(DC*0.86,DC*0.36),constrained_layout=True); a0,a1=ax
pts=np.array([mig.cent_lon,mig.cent_lat]).T.reshape(-1,1,2); segs=np.concatenate([pts[:-1],pts[1:]],axis=1)
lc=LineCollection(segs,cmap='viridis',linewidth=2); lc.set_array(mig.year.values[:-1]); a0.add_collection(lc)
scz=a0.scatter(mig.cent_lon,mig.cent_lat,c=mig.year,cmap='viridis',s=13,zorder=3,edgecolor='white',lw=0.3)
a0.annotate('1985',(mig.cent_lon.iloc[0],mig.cent_lat.iloc[0]),xytext=(0,10),textcoords='offset points',fontsize=6.3,ha='center',va='bottom')
a0.annotate('2020',(mig.cent_lon.iloc[-1],mig.cent_lat.iloc[-1]),xytext=(6,2),textcoords='offset points',fontsize=6.3)
a0.set_xlabel('Centroid longitude (°E)'); a0.set_ylabel('Centroid latitude (°N)'); a0.set_title('Centroid trajectory'); a0.margins(0.12)
cax=a0.inset_axes([0.46,0.90,0.48,0.045]); cb=fig.colorbar(scz,cax=cax,orientation='horizontal',ticks=[1985,2000,2020]); cb.set_label('Year',fontsize=6,labelpad=1); cb.ax.tick_params(labelsize=5.5); plab(a0,'a')
a1.plot(mig.year,mig.cent_lat,'o-',color=OI['blue'],ms=3,mec='white',mew=0.3); s,ic=sen(mig.year,mig.cent_lat); a1.plot([mig.year.min(),mig.year.max()],[ic+s*mig.year.min(),ic+s*mig.year.max()],'--',color='0.35',lw=0.9)
a1.set_ylabel('Centroid latitude (°N)',color=OI['blue']); a1.tick_params(axis='y',colors=OI['blue']); a1.set_xlabel('Year')
a2=a1.twinx(); a2.plot(mig.year,mig.mean_elev,'s-',color=OI['orange'],ms=3,mec='white',mew=0.3); a2.set_ylabel('Centroid elevation (m)',color=OI['orange']); a2.tick_params(axis='y',colors=OI['orange'])
a1.set_title('Northward and downslope migration'); a1.xaxis.set_major_locator(MaxNLocator(5))
a1.text(0.30,0.955,f'Latitude +0.74° (≈82 km) {star(mk_["cent_lat"]["MK_p"])}',transform=a1.transAxes,fontsize=6,color=OI['blue'],ha='left',va='top')
a1.text(0.30,0.085,f'Elevation −99 m {star(mk_["mean_elev"]["MK_p"])}',transform=a1.transAxes,fontsize=6,color=OI['orange'],ha='left',va='bottom'); plab(a1,'b')
sv(fig,'Fig3_centroid_migration'); print('Fig3 ok')

# ===== S2 climate trend =====
fig,ax=plt.subplots(1,2,figsize=(DC*0.66,DC*0.30),constrained_layout=True)
tm=clim.gs_temp.mean(); da=clim.gs_temp-tm
ax[0].bar(clim.year,da,color=[OI['verm'] if v>=0 else OI['blue'] for v in da],width=0.8); s,ic=sen(clim.year,clim.gs_temp); ax[0].plot(clim.year,(ic+s*clim.year)-tm,'-',color='0.2',lw=1.1)
ax[0].axhline(0,color='k',lw=0.5); ax[0].set_ylabel('GS temp. anomaly (°C)'); ax[0].set_title('Growing-season warming'); ax[0].set_ylim(-1.15,0.98); ax[0].text(0.97,0.06,f'+0.35 °C dec$^{{-1}}$ {star(mk_["gs_temp"]["MK_p"])}',transform=ax[0].transAxes,fontsize=6.3,va='bottom',ha='right')
pm=clim.gs_prec.mean(); dp=clim.gs_prec-pm; ax[1].bar(clim.year,dp,color=[OI['blue'] if v>=0 else OI['orange'] for v in dp],width=0.8); s,ic=sen(clim.year,clim.gs_prec); ax[1].plot(clim.year,(ic+s*clim.year)-pm,'-',color='0.2',lw=1.1)
ax[1].axhline(0,color='k',lw=0.5); ax[1].set_ylabel('GS precip. anomaly (mm)'); ax[1].set_title('Growing-season precipitation'); ax[1].set_ylim(-175,235); ax[1].text(0.03,0.06,f'−2.1 mm yr$^{{-1}}$ {star(mk_["gs_prec"]["MK_p"])}',transform=ax[1].transAxes,fontsize=6.3,ha='left',va='bottom')
for a_,Lc in zip(ax,'ab'): a_.set_xlabel('Year'); a_.xaxis.set_major_locator(MaxNLocator(5)); plab(a_,Lc)
sv(fig,'Fig4_climate_trend'); print('S2 climate ok')

# ===== S4 phenology (4 panel box) =====
ph=pd.read_csv(PH+'/phenology_station_year.csv'); smGP=ph.groupby('station').agg(lat=('lat','first'),gp=('gp_sow_mat','mean')).dropna()
cm=pd.read_csv(PH+'/county_at10_gradient.csv'); prov=pd.read_csv(PH+'/province_at10_trend.csv'); at=pd.read_csv(PH+'/phenology_accumulated_temp.csv'); smAT=at.groupby('station').agg(lat=('lat','first'),gp_at10=('gp_at10','mean'))
fig,ax=plt.subplots(2,2,figsize=(DC*0.78,DC*0.62),constrained_layout=True)
a=ax[0,0]; a.scatter(smGP.lat,smGP.gp,s=20,color=OI['green'],edgecolor='white',zorder=3); b=np.polyfit(smGP.lat,smGP.gp,1); xr=np.array([smGP.lat.min(),smGP.lat.max()]); a.plot(xr,np.polyval(b,xr),'--',color='0.3',lw=1)
r=np.corrcoef(smGP.lat,smGP.gp)[0,1]; a.set_xlabel('Station latitude (°N)'); a.set_ylabel('Growing period (days)'); a.set_title('Growing period vs latitude'); a.text(0.05,0.08,f'r = {r:.2f}, n = {len(smGP)}',transform=a.transAxes,fontsize=6.5); plab(a,'a')
a=ax[0,1]; a.scatter(cm.elev_mean,cm.at10,s=16,color=OI['verm'],edgecolor='white',zorder=3); b=np.polyfit(cm.elev_mean,cm.at10,1); xr=np.array([cm.elev_mean.min(),cm.elev_mean.max()]); a.plot(xr,np.polyval(b,xr),'--',color='0.3',lw=1)
r=np.corrcoef(cm.elev_mean,cm.at10)[0,1]; a.set_xlabel('County mean elevation (m)'); a.set_ylabel('GS acc. temp. ≥10°C (°C·d)'); a.set_title('Thermal resource vs elevation'); a.text(0.05,0.08,f'r = {r:.2f}, n = {len(cm)}',transform=a.transAxes,fontsize=6.5); plab(a,'b')
a=ax[1,0]; a.plot(prov.year,prov.at10,'o-',color=OI['orange'],ms=3,mec='white',mew=0.3,zorder=3); s,ic=sen(prov.year,prov.at10); a.plot(prov.year,ic+s*prov.year,'--',color='0.3',lw=1)
a.set_xlabel('Year'); a.set_ylabel('GS acc. temp. ≥10°C (°C·d)'); a.set_title('Rising thermal resource'); a.xaxis.set_major_locator(MaxNLocator(5)); a.text(0.95,0.06,f'+{s*10:.0f} °C·d dec$^{{-1}}$ (Δ≈{prov.at10.values[-1]-prov.at10.values[0]:.0f}) ***',transform=a.transAxes,fontsize=6.3,va='bottom',ha='right'); plab(a,'c')
a=ax[1,1]; a.hist(at.gp_at10,bins=14,color=OI['blue'],alpha=0.85,edgecolor='white'); mval=at.gp_at10.mean(); a.axvline(mval,color='0.2',lw=1.2,ls='--'); a.set_xlabel('Growing-period acc. temp. (°C·d)'); a.set_ylabel('Count (station-years)'); a.set_title('Thermal requirement to maturity'); a.text(0.95,0.9,f'mean ≈ {mval:.0f} °C·d\n(vs latitude r = {np.corrcoef(smAT.lat,smAT.gp_at10)[0,1]:.2f})',transform=a.transAxes,fontsize=6.3,va='top',ha='right'); plab(a,'d')
sv(fig,'FigS_phenology'); print('S4 phenology ok')

# ===== S5 data agreement =====
yb2=pd.read_csv(MID+'/yield_panel_province.csv').set_index('year')['area_ha'].rename('yb'); rsa=pa.groupby('year')['paddy_ha'].sum().rename('rs')
m=pd.concat([rsa,yb2],axis=1).dropna()
fig,ax=plt.subplots(1,2,figsize=(DC*0.72,DC*0.32),constrained_layout=True)
sc=ax[0].scatter(m.yb/1e4,m.rs/1e4,c=m.index,cmap='viridis',s=20,edgecolor='white',lw=0.3); lim=[m.min().min()/1e4*0.9,m.max().max()/1e4*1.05]; ax[0].plot(lim,lim,'--',color='0.5',lw=0.8)
ax[0].set_xlabel('Yearbook sown area (10$^4$ ha)'); ax[0].set_ylabel('Satellite mapped area (10$^4$ ha)'); r=np.corrcoef(m.yb,m.rs)[0,1]; ax[0].set_title('Satellite vs statistical area'); ax[0].text(0.95,0.06,f'r = {r:.2f}, n = {len(m)}\n1:1 line',transform=ax[0].transAxes,fontsize=6.3,va='bottom',ha='right')
cb=fig.colorbar(sc,ax=ax[0],shrink=0.8,pad=0.02,ticks=[1985,2000,2020]); cb.set_label('Year',fontsize=6); plab(ax[0],'a')
rd=100*(m.rs-m.yb)/m.yb; ax[1].axhline(0,color='0.6',lw=0.6); ax[1].bar(m.index,rd,color=[OI['orange'] if v>0 else OI['blue'] for v in rd],width=0.8); ax[1].set_xlabel('Year'); ax[1].set_ylabel('Relative difference (%)'); ax[1].set_title('Convergence over time'); ax[1].xaxis.set_major_locator(MaxNLocator(5)); ax[1].text(0.12,0.06,'large early divergence\n→ <2% by 2020',transform=ax[1].transAxes,fontsize=6,va='bottom',ha='left'); plab(ax[1],'b')
sv(fig,'FigS2_data_agreement'); print('S5 agreement ok')

# ===== S1 maps (字体一致; axis off 无框) =====
fig,axes=plt.subplots(1,3,figsize=(DC,DC*0.30))
vmx=pa[pa.year.isin([1985,2000,2020])]['paddy_ha'].quantile(0.98)/1e4
for ax_,yr,Lc in zip(axes,[1985,2000,2020],'abc'):
    mm=g.merge(pa[pa.year==yr][['CountyCode','paddy_ha']],on='CountyCode',how='left'); mm['pk']=mm['paddy_ha']/1e4
    mm.plot(column='pk',ax=ax_,cmap='YlGnBu',vmin=0,vmax=vmx,edgecolor='0.6',lw=0.2); ax_.set_title(str(yr)); ax_.axis('off'); plab(ax_,Lc)
sm=mpl.cm.ScalarMappable(cmap='YlGnBu',norm=mpl.colors.Normalize(0,vmx)); fig.colorbar(sm,ax=axes,shrink=0.6,label='Paddy area (10$^4$ ha)',pad=0.01)
sv(fig,'Fig6_paddy_expansion_maps'); print('S1 maps ok')
