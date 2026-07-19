# -*- coding: utf-8 -*-
"""Fig4 -> 3面板(去d); Fig5 -> 3面板(a,b加拟合方程+r,p; c=移入的暖月热惩罚PDP)。"""
import json,numpy as np,pandas as pd,warnings; warnings.filterwarnings('ignore')
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import partial_dependence
from scipy.stats import pearsonr
import shap
import matplotlib as mpl; mpl.use('Agg'); import matplotlib.pyplot as plt
mpl.rcParams.update({'font.family':'serif','font.serif':['Times New Roman','Liberation Serif','Nimbus Roman','DejaVu Serif'],'font.weight':'bold','axes.labelweight':'bold','axes.titleweight':'bold','svg.fonttype':'none','pdf.fonttype':42,
 'font.size':7,'axes.titlesize':7.5,'axes.labelsize':7,'xtick.labelsize':6.3,'ytick.labelsize':6.3,'legend.fontsize':6.3,'axes.linewidth':0.8,'legend.frameon':False})
OI={'blue':'#0072B2','green':'#009E73','orange':'#E69F00','verm':'#D55E00','sky':'#56B4E9','grey':'#7A7A7A'}; MM=1/25.4; DC=183*MM
ROOT='/sessions/vibrant-bold-franklin/mnt/水稻'; PROJ=ROOT+'/增温驱动吉林水田扩张与增产_1985-2020'; MID=PROJ+'/02_中间数据'; PUB=PROJ+'/03_figures/pub'
def sv(fig,n): fig.savefig(PUB+'/'+n+'.pdf',bbox_inches='tight'); fig.savefig(PUB+'/'+n+'.png',dpi=600,bbox_inches='tight'); fig.savefig(PUB+'/'+n+'.svg',bbox_inches='tight'); plt.close(fig)
def plab(ax,L): ax.annotate(L,xy=(0,1),xycoords='axes fraction',xytext=(-24,6),textcoords='offset points',fontsize=9,fontweight='bold',va='bottom')
NICE={'gs_tmean':'GS mean T','gdd':'GDD','at10':'AAT≥10°C','t_warm':'Warmest-mo. T','t_may':'May T','prec_gs':'GS precip','prec_78':'Jul–Aug precip','prec_conc':'Precip conc.'}
CLIM=['gs_tmean','gdd','at10','t_warm','t_may','prec_gs','prec_78','prec_conc']; ANOM=[c+'_a' for c in CLIM]
a=json.load(open(MID+'/attribution_results.json'))
pan=pd.read_csv(MID+'/county_year_panel.csv'); pan['CountyCode']=pan['CountyCode'].astype(str)
F=pd.read_csv(MID+'/climate_features_county_year.csv'); F['CountyCode']=F['CountyCode'].astype(str)
df=pan.merge(F,on=['CountyCode','year'],how='left').dropna(subset=['yield_kg_ha']+CLIM); df=df[df.gdd>0].copy(); df['ln_yield']=np.log(df['yield_kg_ha'])
df=df.groupby('CountyCode',group_keys=False).apply(lambda gg: gg.assign(r=(gg['ln_yield']-np.polyval(np.polyfit(gg.year,gg.ln_yield,1),gg.year)) if gg.year.nunique()>=8 else np.nan)).dropna(subset=['r'])
for c in CLIM: df[c+'_a']=df[c]-df.groupby('CountyCode')[c].transform('mean')
X=df[ANOM].values; y=df['r'].values; df['ya']=100*df['r']
rf=RandomForestRegressor(n_estimators=300,min_samples_leaf=5,n_jobs=-1,random_state=42).fit(X,y)
sv_=shap.TreeExplainer(rf).shap_values(X); order=np.argsort(np.abs(sv_).mean(0)); topi=CLIM.index('t_warm')
pdp=partial_dependence(rf,X,[topi],grid_resolution=40); px=np.array(pdp['grid_values'][0]); py=np.array(pdp['average'][0])
meanTw=float(df['t_warm'].mean())
def beeswarm(ax,SV,XX,od,title):
    for yi,fi in enumerate(od):
        s=SV[:,fi]; xv=XX[:,fi]; v=np.clip((xv-np.nanpercentile(xv,5))/(np.nanpercentile(xv,95)-np.nanpercentile(xv,5)+1e-9),0,1)
        j=(np.random.RandomState(fi).rand(len(s))-0.5)*0.34; scp=ax.scatter(s,np.full_like(s,yi)+j,c=v,cmap='coolwarm',s=4.5,alpha=0.75,edgecolor='none',rasterized=True)
    ax.axvline(0,color='0.6',lw=0.6); ax.set_yticks(range(len(od))); ax.set_yticklabels([NICE[CLIM[i]] for i in od]); ax.set_title(title); return scp

# ===== Fig4 attribution: 3 面板 =====
fig=plt.figure(figsize=(DC,DC*0.34)); gs=fig.add_gridspec(1,3,width_ratios=[1,1.35,1.55],wspace=0.6)
axA=fig.add_subplot(gs[0,0]); axB=fig.add_subplot(gs[0,1]); axC=fig.add_subplot(gs[0,2])
axA.bar(['RF','Linear'],[a['yield_climate_share_CV_R2_RF'],a['yield_climate_share_CV_R2_linear']],color=[OI['green'],OI['sky']],width=0.55,edgecolor='white'); axA.set_ylabel('Climate share (CV R²)'); axA.set_title('Climate share of\nyield variability'); axA.set_ylim(0,0.30)
for i,v in enumerate([a['yield_climate_share_CV_R2_RF'],a['yield_climate_share_CV_R2_linear']]): axA.text(i,v+0.008,f'{v:.2f}',ha='center',fontsize=6.5)
plab(axA,'a')
i1=json.load(open(PROJ+'/09_投稿前修订_R1/02_outputs/issue1_attribution_range.json'))
lo,ce,lob,up=i1['lower_pct'],i1['central_pct'],i1['lobell_counterfactual_pct'],i1['upper_pct']
axB.vlines(0,lo,up,color=OI['verm'],lw=7,alpha=0.30)
axB.plot([-.16,.16],[lo,lo],color='0.4',lw=1.1); axB.plot([-.16,.16],[up,up],color='0.4',lw=1.1)
axB.plot(0,ce,'o',color=OI['verm'],ms=8,zorder=4)
axB.plot(0,lob,'s',color=OI['orange'],ms=6,zorder=4)
axB.text(0.18,up,'upper %.1f%%'%up,va='center',fontsize=6)
axB.text(0.18,ce,'central %.1f%%'%ce,va='center',fontsize=6,color=OI['verm'])
axB.text(0.18,lob,'Lobell %.1f%%'%lob,va='center',fontsize=6,color=OI['orange'])
axB.text(0.18,lo,'lower %.1f%%'%lo,va='center',fontsize=6)
axB.set_xlim(-0.5,1.55); axB.set_xticks([]); axB.set_ylim(0,21)
axB.set_ylabel('Climate contribution to\nyield gain (%)'); axB.set_title('Climate share is a\nbounded range')
plab(axB,'b')
scp=beeswarm(axC,sv_,X,order,'Climate drivers of yield (SHAP)'); axC.set_xlabel('SHAP value (→ higher yield)')
cb2=fig.colorbar(scp,ax=axC,shrink=0.8,pad=0.02,ticks=[0,1]); cb2.set_label('Feature value',fontsize=6); cb2.ax.set_yticklabels(['Low','High'],fontsize=5.5); plab(axC,'c')
sv(fig,'Fig8_attribution'); print('Fig4 (3-panel) ok')

# ===== Fig5 temperature response: 3 面板 =====
def binresp(col,step):
    lo,hi=np.floor(df[col].min()),np.ceil(df[col].max()); bins=np.arange(lo,hi+step,step)
    df['_b']=pd.cut(df[col],bins); gg=df.groupby('_b')['ya'].agg(['mean','sem','count']); gg=gg[gg['count']>=15]
    gg['ctr']=[iv.mid for iv in gg.index]; return gg
def fit_eq(gg,deg):
    x=gg.ctr.values.astype(float); ym=gg['mean'].values; w=gg['count'].values
    b=np.polyfit(x,ym,deg,w=np.sqrt(w)); pred=np.polyval(b,x); r,p=pearsonr(ym,pred)
    if deg==1: eq='y = %.2f x %+ .1f'%(b[0],b[1])
    else: eq='y = %.3f x² %+ .2f x %+ .1f'%(b[0],b[1],b[2])
    return b,eq,r,p
fig,ax=plt.subplots(1,3,figsize=(DC,DC*0.33),constrained_layout=True)
# (a) gs_tmean
g0=binresp('gs_tmean',0.5); b0,eq0,r0,p0=fit_eq(g0,1)
ax[0].axhline(0,color='0.6',lw=0.6); ax[0].errorbar(g0.ctr,g0['mean'],yerr=g0['sem'],fmt='o',color=OI['blue'],ms=3.5,mec='white',mew=0.4,capsize=2,lw=0,elinewidth=0.8)
xf=np.linspace(g0.ctr.min(),g0.ctr.max(),100); ax[0].plot(xf,np.polyval(b0,xf),'-',color=OI['blue'],lw=1.3)
ax[0].set_xlabel('Growing-season mean T (°C)'); ax[0].set_ylabel('Climate-driven yield anomaly (%)'); ax[0].set_title('Growing-season mean T')
ax[0].text(0.96,0.05,eq0+'\n'+('r = %.2f, p = %.2g'%(r0,p0)),transform=ax[0].transAxes,fontsize=6.2,va='bottom',ha='right',color=OI['blue']); plab(ax[0],'a')
# (b) t_warm
g1=binresp('t_warm',0.5); b1,eq1,r1,p1=fit_eq(g1,2); opt=-b1[1]/(2*b1[0])
ax[1].axhline(0,color='0.6',lw=0.6); ax[1].errorbar(g1.ctr,g1['mean'],yerr=g1['sem'],fmt='o',color=OI['verm'],ms=3.5,mec='white',mew=0.4,capsize=2,lw=0,elinewidth=0.8)
xf=np.linspace(g1.ctr.min(),g1.ctr.max(),100); ax[1].plot(xf,np.polyval(b1,xf),'-',color=OI['verm'],lw=1.3); ax[1].axvline(opt,color=OI['orange'],lw=1,ls=':')
ax[1].set_xlabel('Warmest-month T (°C)'); ax[1].set_ylabel('Climate-driven yield anomaly (%)'); ax[1].set_title('Warmest-month T: optimum ≈ 22 °C')
ax[1].text(0.985,0.05,eq1+'\n'+('r = %.2f, p = %.2g'%(r1,p1)),transform=ax[1].transAxes,fontsize=5.0,va='bottom',ha='right',color=OI['verm'])
ax[1].text(0.085,0.05,'fit to binned means;\nn = %d, obs. max %.1f'%(len(df),df.t_warm.max())+' \u00b0C',transform=ax[1].transAxes,fontsize=5.0,va='bottom',ha='left',color='0.45')
plab(ax[1],'b')
# (c) RF 偏依赖 暖月热惩罚(绝对温度) + 样本外阴影
ax[2].axvspan(df.t_warm.max(),29.8,color='0.88',alpha=0.5,zorder=0)
ax[2].plot(px+meanTw,100*py,'-',color=OI['verm'],lw=1.6); ax[2].axhline(0,color='0.7',lw=0.5)
ax[2].set_xlabel('Warmest-month T (\u00b0C)'); ax[2].set_ylabel('Partial effect on yield (%)'); ax[2].set_title('Heat penalty (RF partial dependence)')
ax[2].text(0.96,0.94,'declining \u2192\nheat penalty',transform=ax[2].transAxes,fontsize=7.5,va='top',ha='right',color=OI['verm']); plab(ax[2],'c')
sv(fig,'Fig8_temp_response'); print('Fig4/5 revised ok')
