# -*- coding: utf-8 -*-
"""重规划 Fig3/Fig4/Fig8：修文字重叠 + Fig8改2×2。constrained_layout + 偏移面板标签。"""
import json,numpy as np,pandas as pd,warnings; warnings.filterwarnings('ignore')
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import partial_dependence
import shap
import matplotlib as mpl; mpl.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from matplotlib.collections import LineCollection
mpl.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial','DejaVu Sans'],'svg.fonttype':'none','pdf.fonttype':42,
 'font.size':7,'axes.titlesize':8,'axes.labelsize':7,'xtick.labelsize':6.3,'ytick.labelsize':6.3,'legend.fontsize':6.3,
 'axes.linewidth':0.7,'axes.spines.top':False,'axes.spines.right':False,'legend.frameon':False,'lines.linewidth':1.1})
OI={'blue':'#0072B2','orange':'#E69F00','green':'#009E73','verm':'#D55E00','sky':'#56B4E9','grey':'#7A7A7A'}
MM=1/25.4; DC=183*MM
ROOT='/sessions/vibrant-bold-franklin/mnt/水稻'; MID=ROOT+'/增温驱动吉林水田扩张与增产_1985-2020/02_中间数据'; PUB=ROOT+'/增温驱动吉林水田扩张与增产_1985-2020/03_figures/pub'
def save(fig,n): fig.savefig(PUB+'/'+n+'.pdf',bbox_inches='tight'); fig.savefig(PUB+'/'+n+'.png',dpi=600,bbox_inches='tight'); fig.savefig(PUB+'/'+n+'.svg',bbox_inches='tight'); plt.close(fig)
def plab(ax,L): ax.annotate(L,xy=(0,1),xycoords='axes fraction',xytext=(-34,5),textcoords='offset points',fontsize=9,fontweight='bold',va='bottom',ha='left')
def sen(x,y):
    from itertools import combinations; x=np.asarray(x,float);y=np.asarray(y,float);m=~np.isnan(y);x,y=x[m],y[m]
    s=np.median([(y[j]-y[i])/(x[j]-x[i]) for i,j in combinations(range(len(x)),2)]); return s,np.median(y)-s*np.median(x)
def star(p): return '***' if p<0.001 else ('**' if p<0.01 else ('*' if p<0.05 else 'n.s.'))
mk=json.load(open(MID+'/mk_trends.json')); a=json.load(open(MID+'/attribution_results.json'))
mig=pd.read_csv(MID+'/centroid_migration.csv').sort_values('year'); clim=pd.read_csv(MID+'/climate_province.csv').sort_values('year')

# ===================== Fig3 =====================
fig,ax=plt.subplots(1,2,figsize=(DC*0.86,DC*0.36),constrained_layout=True)
a0,a1=ax
pts=np.array([mig.cent_lon,mig.cent_lat]).T.reshape(-1,1,2); segs=np.concatenate([pts[:-1],pts[1:]],axis=1)
lc=LineCollection(segs,cmap='viridis',linewidth=2); lc.set_array(mig.year.values[:-1]); a0.add_collection(lc)
sc=a0.scatter(mig.cent_lon,mig.cent_lat,c=mig.year,cmap='viridis',s=13,zorder=3,edgecolor='white',lw=0.3)
a0.annotate('1985',(mig.cent_lon.iloc[0],mig.cent_lat.iloc[0]),xytext=(6,-2),textcoords='offset points',fontsize=6.3)
a0.annotate('2020',(mig.cent_lon.iloc[-1],mig.cent_lat.iloc[-1]),xytext=(6,2),textcoords='offset points',fontsize=6.3)
a0.set_xlabel('Centroid longitude (°E)'); a0.set_ylabel('Centroid latitude (°N)'); a0.set_title('Centroid trajectory')
a0.margins(0.12)
cax=a0.inset_axes([0.46,0.90,0.48,0.045]); cb=fig.colorbar(sc,cax=cax,orientation='horizontal',ticks=[1985,2000,2020]); cb.set_label('Year',fontsize=6,labelpad=1); cb.ax.tick_params(labelsize=5.5)
plab(a0,'a')
a1.plot(mig.year,mig.cent_lat,'o-',color=OI['blue'],ms=3,mec='white',mew=0.3)
s,ic=sen(mig.year,mig.cent_lat); a1.plot([mig.year.min(),mig.year.max()],[ic+s*mig.year.min(),ic+s*mig.year.max()],'--',color='0.35',lw=0.9)
a1.set_ylabel('Centroid latitude (°N)',color=OI['blue']); a1.tick_params(axis='y',colors=OI['blue']); a1.set_xlabel('Year')
a2=a1.twinx(); a2.plot(mig.year,mig.mean_elev,'s-',color=OI['orange'],ms=3,mec='white',mew=0.3)
a2.set_ylabel('Centroid elevation (m)',color=OI['orange']); a2.tick_params(axis='y',colors=OI['orange']); a2.spines['top'].set_visible(False)
a1.set_title('Northward and downslope migration'); a1.xaxis.set_major_locator(MaxNLocator(5))
# 两条趋势标注放中上空白区(避开数据交叉与坐标)
a1.text(0.30,0.955,f'Latitude +0.74° (≈82 km) {star(mk["cent_lat"]["MK_p"])}',transform=a1.transAxes,fontsize=6,color=OI['blue'],ha='left',va='top')
a1.text(0.30,0.085,f'Elevation −99 m {star(mk["mean_elev"]["MK_p"])}',transform=a1.transAxes,fontsize=6,color=OI['orange'],ha='left',va='bottom')
plab(a1,'b')
save(fig,'Fig3_centroid_migration'); print('Fig3 done')

# ===================== Fig4 =====================
fig,ax=plt.subplots(1,2,figsize=(DC*0.74,DC*0.34),constrained_layout=True)
tm=clim.gs_temp.mean(); da=clim.gs_temp-tm
ax[0].bar(clim.year,da,color=[OI['verm'] if v>=0 else OI['blue'] for v in da],width=0.8)
s,ic=sen(clim.year,clim.gs_temp); ax[0].plot(clim.year,(ic+s*clim.year)-tm,'-',color='0.2',lw=1.1)
ax[0].axhline(0,color='k',lw=0.5); ax[0].set_ylabel('GS temp. anomaly (°C)'); ax[0].set_title('Growing-season warming')
ax[0].set_ylim(-1.15,0.98); ax[0].text(0.03,0.06,f'+0.35 °C dec$^{{-1}}$ {star(mk["gs_temp"]["MK_p"])}',transform=ax[0].transAxes,fontsize=6.3,va='bottom')
pm=clim.gs_prec.mean(); dp=clim.gs_prec-pm
ax[1].bar(clim.year,dp,color=[OI['blue'] if v>=0 else OI['orange'] for v in dp],width=0.8)
s,ic=sen(clim.year,clim.gs_prec); ax[1].plot(clim.year,(ic+s*clim.year)-pm,'-',color='0.2',lw=1.1)
ax[1].axhline(0,color='k',lw=0.5); ax[1].set_ylabel('GS precip. anomaly (mm)'); ax[1].set_title('Growing-season precipitation')
ax[1].set_ylim(-175,235); ax[1].text(0.97,0.06,f'−2.1 mm yr$^{{-1}}$ {star(mk["gs_prec"]["MK_p"])}',transform=ax[1].transAxes,fontsize=6.3,ha='right',va='bottom')
for a_,L in zip(ax,'ab'): a_.set_xlabel('Year'); a_.xaxis.set_major_locator(MaxNLocator(5)); plab(a_,L)
save(fig,'Fig4_climate_trend'); print('Fig4 done')

# ===================== Fig8 (2x2) =====================
NICE={'gs_tmean':'GS mean T','gdd':'GDD','at10':'AAT≥10°C','t_warm':'Warmest-mo. T','t_may':'May T','prec_gs':'GS precip','prec_78':'Jul–Aug precip','prec_conc':'Precip conc.'}
CLIM=['gs_tmean','gdd','at10','t_warm','t_may','prec_gs','prec_78','prec_conc']; ANOM=[c+'_a' for c in CLIM]
pan=pd.read_csv(MID+'/county_year_panel.csv'); pan['CountyCode']=pan['CountyCode'].astype(str)
F=pd.read_csv(MID+'/climate_features_county_year.csv'); F['CountyCode']=F['CountyCode'].astype(str)
df=pan.merge(F,on=['CountyCode','year'],how='left').dropna(subset=['yield_kg_ha']+CLIM); df=df[df.gdd>0].copy(); df['ln_yield']=np.log(df['yield_kg_ha'])
df=df.groupby('CountyCode',group_keys=False).apply(lambda g: g.assign(r=(g['ln_yield']-np.polyval(np.polyfit(g.year,g.ln_yield,1),g.year)) if g.year.nunique()>=8 else np.nan)).dropna(subset=['r'])
for c in CLIM: df[c+'_a']=df[c]-df.groupby('CountyCode')[c].transform('mean')
X=df[ANOM].values; y=df['r'].values
rf=RandomForestRegressor(n_estimators=300,min_samples_leaf=5,n_jobs=-1,random_state=42).fit(X,y)
sv=shap.TreeExplainer(rf).shap_values(X); order=np.argsort(np.abs(sv).mean(0)); topi=int(np.argmax(np.abs(sv).mean(0))); top=CLIM[topi]
pdp=partial_dependence(rf,X,[topi],grid_resolution=40); px=np.array(pdp['grid_values'][0]); py=np.array(pdp['average'][0])
fig,ax=plt.subplots(2,2,figsize=(DC*0.82,DC*0.66),constrained_layout=True)
axA,axB,axC,axD=ax[0,0],ax[0,1],ax[1,0],ax[1,1]
axA.bar(['RF','Linear'],[a['yield_climate_share_CV_R2_RF'],a['yield_climate_share_CV_R2_linear']],color=[OI['green'],OI['sky']],width=0.55,edgecolor='white')
axA.set_ylabel('Climate share (CV R²)'); axA.set_title('Climate share of yield variability'); axA.set_ylim(0,0.30)
for i,v in enumerate([a['yield_climate_share_CV_R2_RF'],a['yield_climate_share_CV_R2_linear']]): axA.text(i,v+0.008,f'{v:.2f}',ha='center',fontsize=6.5)
plab(axA,'a')
axB.bar(['Technology','Climate'],[a['yield_trend_tech_pct35'],a['yield_trend_climate_pct35']],color=[OI['orange'],OI['verm']],width=0.55,edgecolor='white')
axB.set_ylabel('Yield change 1985–2020 (%)'); axB.set_title('Attribution of yield trend'); axB.set_ylim(0,28)
for i,v in enumerate([a['yield_trend_tech_pct35'],a['yield_trend_climate_pct35']]): axB.text(i,v+0.6,f'{v:.1f}%',ha='center',fontsize=6.5)
plab(axB,'b')
for yi,fi in enumerate(order):
    s=sv[:,fi]; xv=X[:,fi]; v=np.clip((xv-np.nanpercentile(xv,5))/(np.nanpercentile(xv,95)-np.nanpercentile(xv,5)+1e-9),0,1)
    j=(np.random.RandomState(fi).rand(len(s))-0.5)*0.34
    scp=axC.scatter(s,np.full_like(s,yi)+j,c=v,cmap='coolwarm',s=4.5,alpha=0.75,edgecolor='none',rasterized=True)
axC.axvline(0,color='0.6',lw=0.6); axC.set_yticks(range(len(order))); axC.set_yticklabels([NICE[CLIM[i]] for i in order])
axC.set_xlabel('SHAP value (→ higher yield)'); axC.set_title('Climate drivers of yield (SHAP)')
cb2=fig.colorbar(scp,ax=axC,shrink=0.7,pad=0.02,ticks=[0,1]); cb2.set_label('Feature value',fontsize=6); cb2.ax.set_yticklabels(['Low','High'],fontsize=5.5)
plab(axC,'c')
axD.plot(px,py,'-',color=OI['verm'],lw=1.4); axD.axhline(0,color='0.7',lw=0.5)
axD.plot(X[:,topi],np.full(len(X),axD.get_ylim()[0]),'|',color='0.5',ms=4,alpha=0.25)
axD.set_xlabel(NICE[top]+' anomaly (°C)'); axD.set_ylabel('Partial effect on ln(yield)'); axD.set_title('Warmest-month heat penalty')
plab(axD,'d')
save(fig,'Fig8_attribution'); print('Fig8 done (2x2), top=',top)
