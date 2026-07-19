# -*- coding: utf-8 -*-
"""Fig8(归因+SHAP beeswarm,修重叠), Fig9(扩张beeswarm), FigS1(修重叠)。"""
import json,numpy as np,pandas as pd,warnings; warnings.filterwarnings('ignore')
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import partial_dependence
import shap
import matplotlib as mpl; mpl.use('Agg'); import matplotlib.pyplot as plt
mpl.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial','DejaVu Sans'],'svg.fonttype':'none','pdf.fonttype':42,
 'font.size':7,'axes.titlesize':8,'axes.labelsize':7,'xtick.labelsize':6.5,'ytick.labelsize':6.5,'legend.fontsize':6.3,
 'axes.linewidth':0.7,'axes.spines.top':False,'axes.spines.right':False,'legend.frameon':False})
OI={'blue':'#0072B2','orange':'#E69F00','green':'#009E73','verm':'#D55E00','sky':'#56B4E9','grey':'#7A7A7A'}
MM=1/25.4; SC=89*MM; DC=183*MM
ROOT='/sessions/vibrant-bold-franklin/mnt/水稻'; MID=ROOT+'/增温驱动吉林水田扩张与增产_1985-2020/02_中间数据'; PUB=ROOT+'/增温驱动吉林水田扩张与增产_1985-2020/03_figures/pub'
def save(fig,n): fig.savefig(PUB+'/'+n+'.pdf',bbox_inches='tight'); fig.savefig(PUB+'/'+n+'.png',dpi=600,bbox_inches='tight'); fig.savefig(PUB+'/'+n+'.svg',bbox_inches='tight'); plt.close(fig)
def PL(a,L,x=-0.18): a.text(x,1.06,L,transform=a.transAxes,fontsize=9,fontweight='bold',va='top')
NICE={'gs_tmean':'GS mean T','gdd':'GDD','at10':'AAT≥10°C','t_warm':'Warmest-mo. T','t_may':'May T','prec_gs':'GS precip','prec_78':'Jul–Aug precip','prec_conc':'Precip conc.'}
CLIM=['gs_tmean','gdd','at10','t_warm','t_may','prec_gs','prec_78','prec_conc']; ANOM=[c+'_a' for c in CLIM]
a=json.load(open(MID+'/attribution_results.json'))
pan=pd.read_csv(MID+'/county_year_panel.csv'); pan['CountyCode']=pan['CountyCode'].astype(str)
F=pd.read_csv(MID+'/climate_features_county_year.csv'); F['CountyCode']=F['CountyCode'].astype(str)
df=pan.merge(F,on=['CountyCode','year'],how='left').dropna(subset=['yield_kg_ha']+CLIM); df=df[df.gdd>0].copy(); df['ln_yield']=np.log(df['yield_kg_ha'])
def detr(g):
    if g.year.nunique()<8: g['r']=np.nan; return g
    g['r']=g['ln_yield']-np.polyval(np.polyfit(g.year,g.ln_yield,1),g.year); return g
df=df.groupby('CountyCode',group_keys=False).apply(detr).dropna(subset=['r'])
for c in CLIM: df[c+'_a']=df[c]-df.groupby('CountyCode')[c].transform('mean')
X=df[ANOM].values; y=df['r'].values
rf=RandomForestRegressor(n_estimators=300,min_samples_leaf=5,n_jobs=-1,random_state=42).fit(X,y)
sv=shap.TreeExplainer(rf).shap_values(X)
order=np.argsort(np.abs(sv).mean(0))  # ascending
top=CLIM[int(np.argmax(np.abs(sv).mean(0)))]
pdp=partial_dependence(rf,X,[CLIM.index(top)],grid_resolution=40); px=np.array(pdp['grid_values'][0]); py=np.array(pdp['average'][0])

def beeswarm(ax,sv,X,order,title):
    for yi,fi in enumerate(order):
        s=sv[:,fi]; xv=X[:,fi]
        # color by normalized feature value
        v=(xv-np.nanpercentile(xv,5))/(np.nanpercentile(xv,95)-np.nanpercentile(xv,5)+1e-9); v=np.clip(v,0,1)
        jitter=(np.random.RandomState(fi).rand(len(s))-0.5)*0.32
        sc=ax.scatter(s,np.full_like(s,yi)+jitter,c=v,cmap='coolwarm',s=4,alpha=0.7,edgecolor='none',rasterized=True)
    ax.axvline(0,color='0.6',lw=0.6); ax.set_yticks(range(len(order))); ax.set_yticklabels([NICE[CLIM[i]] for i in order])
    ax.set_xlabel('SHAP value (→ higher yield resid.)'); ax.set_title(title)
    return sc

# ===== Fig8 =====
fig=plt.figure(figsize=(DC,DC*0.46)); gs=fig.add_gridspec(2,2,width_ratios=[1,1.35],height_ratios=[1,1],hspace=0.55,wspace=0.4)
axA=fig.add_subplot(gs[0,0]); axB=fig.add_subplot(gs[1,0]); axC=fig.add_subplot(gs[:,1])
axA.bar(['RF','Linear'],[a['yield_climate_share_CV_R2_RF'],a['yield_climate_share_CV_R2_linear']],color=[OI['green'],OI['sky']],width=0.6,edgecolor='white')
axA.set_ylabel('Climate share (CV R²)'); axA.set_title('Climate share of yield variability'); axA.set_ylim(0,0.32)
for i,v in enumerate([a['yield_climate_share_CV_R2_RF'],a['yield_climate_share_CV_R2_linear']]): axA.text(i,v+0.008,f'{v:.2f}',ha='center',fontsize=6.5)
PL(axA,'a',-0.32)
axB.bar(['Technology','Climate'],[a['yield_trend_tech_pct35'],a['yield_trend_climate_pct35']],color=[OI['orange'],OI['verm']],width=0.6,edgecolor='white')
axB.set_ylabel('Yield change\n1985–2020 (%)'); axB.set_title('Attribution of yield trend')
for i,v in enumerate([a['yield_trend_tech_pct35'],a['yield_trend_climate_pct35']]): axB.text(i,v+0.5,f'{v:.1f}%',ha='center',fontsize=6.5)
PL(axB,'b',-0.32)
sc=beeswarm(axC,sv,X,order,'Climate drivers of yield (SHAP)'); PL(axC,'c',-0.28)
cb=fig.colorbar(sc,ax=axC,shrink=0.55,pad=0.02,ticks=[0,1]); cb.set_label('Feature value',fontsize=6.2); cb.ax.set_yticklabels(['Low','High'])
# inset PDP in axC top-left? make small inset
axd=axC.inset_axes([0.5,0.06,0.46,0.34]); axd.plot(px,py,'-',color=OI['verm'],lw=1.2); axd.axhline(0,color='0.7',lw=0.5)
axd.set_title('PDP: '+NICE[top],fontsize=6); axd.tick_params(labelsize=5.5); axd.set_xlabel('anom (°C)',fontsize=5.5); axd.set_ylabel('Δln(yield)',fontsize=5.5)
save(fig,'Fig8_attribution'); print('Fig8 done, top factor',top)

# ===== Fig9: 扩张 beeswarm =====
de=pan.merge(F,on=['CountyCode','year'],how='left').dropna(subset=['paddy_ha']+CLIM); de=de[(de.paddy_ha>0)&(de.gdd>0)].copy(); de['ln_paddy']=np.log(de['paddy_ha'])
for c in CLIM: de[c+'_a']=de[c]-de.groupby('CountyCode')[c].transform('mean')
de['ln_paddy_dm']=de['ln_paddy']-de.groupby('CountyCode')['ln_paddy'].transform('mean')
Xe=de[ANOM].values; ye=de['ln_paddy_dm'].values
rfe=RandomForestRegressor(n_estimators=300,min_samples_leaf=5,n_jobs=-1,random_state=42).fit(Xe,ye)
sve=shap.TreeExplainer(rfe).shap_values(Xe); oe=np.argsort(np.abs(sve).mean(0))
fig,ax=plt.subplots(figsize=(DC*0.5,DC*0.34)); sc=beeswarm(ax,sve,Xe,oe,f"Climate drivers of expansion (CV R²={a['expansion_climate_CV_R2']:.2f})")
cb=fig.colorbar(sc,ax=ax,shrink=0.6,pad=0.02,ticks=[0,1]); cb.set_label('Feature value',fontsize=6.2); cb.ax.set_yticklabels(['Low','High'])
fig.tight_layout(); save(fig,'Fig9_expansion_drivers'); print('Fig9 done')

# ===== FigS1: 修重叠 =====
import pandas as pd
PH=ROOT+'/增温驱动吉林水田扩张与增产_1985-2020/06_phenology'
ph=pd.read_csv(PH+'/phenology_station_year.csv')
# GP-lat 用产量表(更全) 复算
y=pd.read_excel(ROOT+'水稻物候/中国水稻产量.xlsx') if False else None
sm=ph.groupby('station').agg(lat=('lat','first'),gp=('gp_sow_mat','mean')).dropna()
r=np.corrcoef(sm['lat'],sm['gp'])[0,1]
cal={k:ph[k].mean() for k in ['sowing','transplant','heading','maturity']}
fig,ax=plt.subplots(1,2,figsize=(DC*0.7,DC*0.30))
ax[0].scatter(sm['lat'],sm['gp'],s=20,color=OI['green'],edgecolor='white',zorder=3)
b=np.polyfit(sm['lat'],sm['gp'],1); xr=np.array([sm['lat'].min(),sm['lat'].max()]); ax[0].plot(xr,np.polyval(b,xr),'--',color='0.3',lw=1)
ax[0].set_xlabel('Station latitude (°N)'); ax[0].set_ylabel('Growing period (days)'); ax[0].set_title('Thermal constraint on growing period')
ax[0].text(0.05,0.08,f'r = {r:.2f}, n = {len(sm)}',transform=ax[0].transAxes,fontsize=7); PL(ax[0],'a')
# 甘特式物候历: 单条生育期bar + 阶段标记(标签交错,避免重叠)
ks=['sowing','transplant','heading','maturity']; labs=['Sowing','Transplant','Heading','Maturity']; vals=[cal[k] for k in ks]
ax[1].barh(0,vals[-1]-vals[0],left=vals[0],height=0.28,color='#DDE6EF',edgecolor=OI['blue'],lw=0.8,zorder=1)
for i,(v,l) in enumerate(zip(vals,labs)):
    ax[1].plot([v],[0],'o',color=OI['blue'],ms=6,zorder=3,mec='white')
    dy=0.33 if i%2==0 else -0.33; va='bottom' if i%2==0 else 'top'
    ax[1].annotate(f'{l}\nDOY {v:.0f}',(v,0),xytext=(v,dy),ha='center',va=va,fontsize=6.3,
        arrowprops=dict(arrowstyle='-',color='0.6',lw=0.5))
ax[1].set_ylim(-0.9,0.9); ax[1].set_yticks([]); ax[1].set_xlabel('Day of year'); ax[1].set_xlim(90,290)
ax[1].set_title('Mean phenological calendar'); ax[1].spines['left'].set_visible(False); PL(ax[1],'b')
fig.tight_layout(); save(fig,'FigS_phenology'); print('FigS done, GP-lat r=%.2f'%r)
