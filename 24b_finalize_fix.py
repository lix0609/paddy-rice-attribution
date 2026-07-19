# -*- coding: utf-8 -*-
import json,numpy as np,pandas as pd,geopandas as gpd,warnings; warnings.filterwarnings('ignore')
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import partial_dependence
import shap
import matplotlib as mpl; mpl.use('Agg'); import matplotlib.pyplot as plt
mpl.rcParams.update({'font.family':'serif','font.serif':['Times New Roman','Liberation Serif','Nimbus Roman','DejaVu Serif'],'font.weight':'bold','axes.labelweight':'bold','axes.titleweight':'bold','svg.fonttype':'none','pdf.fonttype':42,
 'font.size':7,'axes.titlesize':7.5,'axes.labelsize':7,'xtick.labelsize':6.3,'ytick.labelsize':6.3,'legend.fontsize':6.3,'axes.linewidth':0.7,'legend.frameon':False})
OI={'blue':'#0072B2','green':'#009E73','orange':'#E69F00','verm':'#D55E00','sky':'#56B4E9','grey':'#7A7A7A'}; MM=1/25.4; DC=183*MM
ROOT='/sessions/vibrant-bold-franklin/mnt/水稻'; PROJ=ROOT+'/增温驱动吉林水田扩张与增产_1985-2020'; MID=PROJ+'/02_中间数据'; PUB=PROJ+'/03_figures/pub'; SHP=ROOT+'/吉林省_区县行政边界/吉林省_区县_合并50.shp'
def sv(fig,n): fig.savefig(PUB+'/'+n+'.pdf',bbox_inches='tight'); fig.savefig(PUB+'/'+n+'.png',dpi=600,bbox_inches='tight'); fig.savefig(PUB+'/'+n+'.svg',bbox_inches='tight'); plt.close(fig)
def plab(ax,L,x=-0.2): ax.annotate(L,xy=(0,1),xycoords='axes fraction',xytext=(x*100*0.32,6),textcoords='offset points',fontsize=9,fontweight='bold',va='bottom')
NICE={'gs_tmean':'GS mean T','gdd':'GDD','at10':'AAT≥10°C','t_warm':'Warmest-mo. T','t_may':'May T','prec_gs':'GS precip','prec_78':'Jul–Aug precip','prec_conc':'Precip conc.'}
CLIM=['gs_tmean','gdd','at10','t_warm','t_may','prec_gs','prec_78','prec_conc']; ANOM=[c+'_a' for c in CLIM]
a=json.load(open(MID+'/attribution_results.json')); rr=json.load(open(MID+'/regression_results.json'))
pan=pd.read_csv(MID+'/county_year_panel.csv'); pan['CountyCode']=pan['CountyCode'].astype(str)
F=pd.read_csv(MID+'/climate_features_county_year.csv'); F['CountyCode']=F['CountyCode'].astype(str)
df=pan.merge(F,on=['CountyCode','year'],how='left').dropna(subset=['yield_kg_ha']+CLIM); df=df[df.gdd>0].copy(); df['ln_yield']=np.log(df['yield_kg_ha'])
df=df.groupby('CountyCode',group_keys=False).apply(lambda gg: gg.assign(r=(gg['ln_yield']-np.polyval(np.polyfit(gg.year,gg.ln_yield,1),gg.year)) if gg.year.nunique()>=8 else np.nan)).dropna(subset=['r'])
for c in CLIM: df[c+'_a']=df[c]-df.groupby('CountyCode')[c].transform('mean')
X=df[ANOM].values; y=df['r'].values; df['ya']=100*df['r']
rf=RandomForestRegressor(n_estimators=300,min_samples_leaf=5,n_jobs=-1,random_state=42).fit(X,y)
sv_=shap.TreeExplainer(rf).shap_values(X); order=np.argsort(np.abs(sv_).mean(0)); topi=int(np.argmax(np.abs(sv_).mean(0))); top=CLIM[topi]
pdp=partial_dependence(rf,X,[topi],grid_resolution=40); px=np.array(pdp['grid_values'][0]); py=np.array(pdp['average'][0])
def beeswarm(ax,SV,XX,od,title):
    for yi,fi in enumerate(od):
        s=SV[:,fi]; xv=XX[:,fi]; v=np.clip((xv-np.nanpercentile(xv,5))/(np.nanpercentile(xv,95)-np.nanpercentile(xv,5)+1e-9),0,1)
        j=(np.random.RandomState(fi).rand(len(s))-0.5)*0.34; scp=ax.scatter(s,np.full_like(s,yi)+j,c=v,cmap='coolwarm',s=4.5,alpha=0.75,edgecolor='none',rasterized=True)
    ax.axvline(0,color='0.6',lw=0.6); ax.set_yticks(range(len(od))); ax.set_yticklabels([NICE[CLIM[i]] for i in od]); ax.set_title(title); return scp

# ===== Fig4 attribution (2x2, 全框) =====
fig,ax=plt.subplots(2,2,figsize=(DC*0.82,DC*0.66),constrained_layout=True); axA,axB,axC,axD=ax[0,0],ax[0,1],ax[1,0],ax[1,1]
axA.bar(['RF','Linear'],[a['yield_climate_share_CV_R2_RF'],a['yield_climate_share_CV_R2_linear']],color=[OI['green'],OI['sky']],width=0.55,edgecolor='white'); axA.set_ylabel('Climate share (CV R²)'); axA.set_title('Climate share of yield variability'); axA.set_ylim(0,0.30)
for i,v in enumerate([a['yield_climate_share_CV_R2_RF'],a['yield_climate_share_CV_R2_linear']]): axA.text(i,v+0.008,f'{v:.2f}',ha='center',fontsize=6.5)
plab(axA,'a')
axB.bar(['Technology','Climate'],[a['yield_trend_tech_pct35'],a['yield_trend_climate_pct35']],color=[OI['orange'],OI['verm']],width=0.55,edgecolor='white'); axB.set_ylabel('Yield change 1985–2020 (%)'); axB.set_title('Attribution of yield trend'); axB.set_ylim(0,28)
for i,v in enumerate([a['yield_trend_tech_pct35'],a['yield_trend_climate_pct35']]): axB.text(i,v+0.6,f'{v:.1f}%',ha='center',fontsize=6.5)
plab(axB,'b')
scp=beeswarm(axC,sv_,X,order,'Climate drivers of yield (SHAP)'); axC.set_xlabel('SHAP value (→ higher yield)'); cb2=fig.colorbar(scp,ax=axC,shrink=0.7,pad=0.02,ticks=[0,1]); cb2.set_label('Feature value',fontsize=6); cb2.ax.set_yticklabels(['Low','High'],fontsize=5.5); plab(axC,'c')
axD.plot(px,py,'-',color=OI['verm'],lw=1.4); axD.axhline(0,color='0.7',lw=0.5); axD.plot(X[:,topi],np.full(len(X),axD.get_ylim()[0]),'|',color='0.5',ms=4,alpha=0.25); axD.set_xlabel(NICE[top]+' anomaly (°C)'); axD.set_ylabel('Partial effect on ln(yield)'); axD.set_title('Warmest-month heat penalty'); plab(axD,'d')
sv(fig,'Fig8_attribution'); print('Fig4 attribution ok')

# ===== Fig5 temperature response (全框 + 文字重新定位) =====
def binresp(col,step):
    lo,hi=np.floor(df[col].min()),np.ceil(df[col].max()); bins=np.arange(lo,hi+step,step); df['_b']=pd.cut(df[col],bins); gg=df.groupby('_b')['ya'].agg(['mean','sem','count']); gg=gg[gg['count']>=15]; gg['ctr']=[iv.mid for iv in gg.index]; return gg
fig,ax=plt.subplots(1,2,figsize=(DC*0.74,DC*0.34),constrained_layout=True)
g0=binresp('gs_tmean',0.5); ax[0].axhline(0,color='0.6',lw=0.6); ax[0].errorbar(g0.ctr,g0['mean'],yerr=g0['sem'],fmt='o-',color=OI['blue'],ms=3.5,mec='white',mew=0.4,capsize=2,lw=1)
xf=np.linspace(g0.ctr.min(),g0.ctr.max(),100); b=np.polyfit(df['gs_tmean'],df['ya'],2); ax[0].plot(xf,np.polyval(b,xf),'--',color='0.4',lw=1); ax[0].set_xlabel('Growing-season mean temperature (°C)'); ax[0].set_ylabel('Climate-driven yield anomaly (%)'); ax[0].set_title('Growing-season mean T')
ax[0].text(0.97,0.06,'Higher T → higher yield\n(cold-region benefit)',transform=ax[0].transAxes,fontsize=6,va='bottom',ha='right',color=OI['blue']); plab(ax[0],'a')
g1=binresp('t_warm',0.5); ax[1].axhline(0,color='0.6',lw=0.6); ax[1].errorbar(g1.ctr,g1['mean'],yerr=g1['sem'],fmt='o-',color=OI['verm'],ms=3.5,mec='white',mew=0.4,capsize=2,lw=1)
b=np.polyfit(df['t_warm'],df['ya'],2); opt=-b[1]/(2*b[0]); xf=np.linspace(g1.ctr.min(),g1.ctr.max(),100); ax[1].plot(xf,np.polyval(b,xf),'--',color='0.4',lw=1); ax[1].axvline(opt,color=OI['orange'],lw=1,ls=':')
ax[1].set_xlabel('Warmest-month temperature (°C)'); ax[1].set_ylabel('Climate-driven yield anomaly (%)'); ax[1].set_title('Warmest-month T: optimum')
ax[1].text(0.97,0.06,f'optimum ≈ {opt:.1f} °C;\nyield falls beyond ~24 °C',transform=ax[1].transAxes,fontsize=6,va='bottom',ha='right',color=OI['verm']); plab(ax[1],'b')
sv(fig,'Fig8_temp_response'); print('Fig5 temp ok (opt %.1f)'%opt)

# ===== Fig6 frontier (面板b棒棒糖标注上移) =====
g=gpd.read_file(SHP).to_crs(4326); g['CountyCode']=g['CountyCode'].astype(str)
cc=pd.read_csv(MID+'/county_change_summary.csv'); cc['CountyCode']=cc['CountyCode'].astype(str); bands=pd.DataFrame(rr['latitude_bands']).set_index('band').loc[['South','Central','North']]
m=g.merge(cc[['CountyCode','expansion_x']],on='CountyCode',how='left'); m['ec']=m['expansion_x'].clip(upper=8)
fig,ax=plt.subplots(1,2,figsize=(DC*0.72,DC*0.32))
m.plot(column='ec',ax=ax[0],cmap='YlOrRd',legend=True,edgecolor='0.5',lw=0.25,legend_kwds={'label':'Expansion factor (capped 8)','shrink':0.68}); ax[0].set_title('County expansion factor'); ax[0].axis('off'); plab(ax[0],'a')
yb2=np.arange(3); ex=bands['expansion_x'].values; wm=bands['warming_C_dec'].values
ax[1].hlines(yb2,1,ex,color=OI['grey'],lw=1.4,zorder=1); scc=ax[1].scatter(ex,yb2,c=wm,cmap='OrRd',s=90,zorder=3,edgecolor='k',lw=0.5,vmin=0.32,vmax=0.37)
for i,(e,w) in enumerate(zip(ex,wm)): ax[1].text(e,i+0.30,f'{e:.1f}×',va='bottom',ha='center',fontsize=7)  # 上移到点上方
ax[1].set_yticks(yb2); ax[1].set_yticklabels(bands.index); ax[1].set_xlabel('Expansion factor (2020/1985)'); ax[1].set_xlim(0.8,6.4); ax[1].set_ylim(-0.5,2.6); ax[1].set_title('Expansion by latitude band'); ax[1].axvline(1,color='0.7',lw=0.6,ls=':')
cb=fig.colorbar(scc,ax=ax[1],shrink=0.7,pad=0.02); cb.set_label('Warming (°C dec$^{-1}$)',fontsize=6.3); plab(ax[1],'b')
fig.tight_layout(); sv(fig,'Fig7_expansion_vs_warming'); print('Fig6 frontier ok')

# ===== S3 drivers beeswarm (全框) =====
de=pan.merge(F,on=['CountyCode','year'],how='left').dropna(subset=['paddy_ha']+CLIM); de=de[(de.paddy_ha>0)&(de.gdd>0)].copy(); de['ln_paddy']=np.log(de['paddy_ha'])
for c in CLIM: de[c+'_a']=de[c]-de.groupby('CountyCode')[c].transform('mean')
de['ln_paddy_dm']=de['ln_paddy']-de.groupby('CountyCode')['ln_paddy'].transform('mean')
Xe=de[ANOM].values; ye=de['ln_paddy_dm'].values; rfe=RandomForestRegressor(n_estimators=300,min_samples_leaf=5,n_jobs=-1,random_state=42).fit(Xe,ye); sve=shap.TreeExplainer(rfe).shap_values(Xe); oe=np.argsort(np.abs(sve).mean(0))
fig,ax=plt.subplots(figsize=(DC*0.5,DC*0.34)); scp=beeswarm(ax,sve,Xe,oe,f"Climate drivers of expansion (CV R²={a['expansion_climate_CV_R2']:.2f})"); ax.set_xlabel('SHAP value'); cb=fig.colorbar(scp,ax=ax,shrink=0.6,pad=0.02,ticks=[0,1]); cb.set_label('Feature value',fontsize=6.2); cb.ax.set_yticklabels(['Low','High'],fontsize=5.5); fig.tight_layout(); sv(fig,'Fig9_expansion_drivers'); print('S3 drivers ok')
