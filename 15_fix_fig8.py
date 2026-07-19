import json,numpy as np,pandas as pd,warnings; warnings.filterwarnings('ignore')
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import partial_dependence
import shap
import matplotlib as mpl; mpl.use('Agg'); import matplotlib.pyplot as plt
mpl.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial','DejaVu Sans'],'svg.fonttype':'none','pdf.fonttype':42,
 'font.size':7,'axes.titlesize':7.5,'axes.labelsize':7,'xtick.labelsize':6,'ytick.labelsize':6.3,'axes.linewidth':0.7,
 'axes.spines.top':False,'axes.spines.right':False,'legend.frameon':False})
OI={'green':'#009E73','sky':'#56B4E9','orange':'#E69F00','verm':'#D55E00'}; MM=1/25.4; DC=183*MM
ROOT='/sessions/vibrant-bold-franklin/mnt/水稻'; MID=ROOT+'/增温驱动吉林水田扩张与增产_1985-2020/02_中间数据'; PUB=ROOT+'/增温驱动吉林水田扩张与增产_1985-2020/03_figures/pub'
NICE={'gs_tmean':'GS mean T','gdd':'GDD','at10':'AAT≥10°C','t_warm':'Warmest-mo. T','t_may':'May T','prec_gs':'GS precip','prec_78':'Jul–Aug precip','prec_conc':'Precip conc.'}
CLIM=['gs_tmean','gdd','at10','t_warm','t_may','prec_gs','prec_78','prec_conc']; ANOM=[c+'_a' for c in CLIM]
a=json.load(open(MID+'/attribution_results.json'))
pan=pd.read_csv(MID+'/county_year_panel.csv'); pan['CountyCode']=pan['CountyCode'].astype(str)
F=pd.read_csv(MID+'/climate_features_county_year.csv'); F['CountyCode']=F['CountyCode'].astype(str)
df=pan.merge(F,on=['CountyCode','year'],how='left').dropna(subset=['yield_kg_ha']+CLIM); df=df[df.gdd>0].copy(); df['ln_yield']=np.log(df['yield_kg_ha'])
df=df.groupby('CountyCode',group_keys=False).apply(lambda g: g.assign(r=(g['ln_yield']-np.polyval(np.polyfit(g.year,g.ln_yield,1),g.year)) if g.year.nunique()>=8 else np.nan)).dropna(subset=['r'])
for c in CLIM: df[c+'_a']=df[c]-df.groupby('CountyCode')[c].transform('mean')
X=df[ANOM].values; y=df['r'].values
rf=RandomForestRegressor(n_estimators=300,min_samples_leaf=5,n_jobs=-1,random_state=42).fit(X,y)
sv=shap.TreeExplainer(rf).shap_values(X); order=np.argsort(np.abs(sv).mean(0)); topi=int(np.argmax(np.abs(sv).mean(0))); top=CLIM[topi]
pdp=partial_dependence(rf,X,[topi],grid_resolution=40); px=np.array(pdp['grid_values'][0]); py=np.array(pdp['average'][0])
def PL(ax,L): ax.text(-0.28,1.07,L,transform=ax.transAxes,fontsize=9,fontweight='bold',va='top')
fig=plt.figure(figsize=(DC,DC*0.32)); gs=fig.add_gridspec(1,4,width_ratios=[0.8,0.8,2.0,1.25],wspace=0.95)
axA=fig.add_subplot(gs[0]); axB=fig.add_subplot(gs[1]); axC=fig.add_subplot(gs[2]); axD=fig.add_subplot(gs[3])
axA.bar(['RF','Lin.'],[a['yield_climate_share_CV_R2_RF'],a['yield_climate_share_CV_R2_linear']],color=[OI['green'],OI['sky']],width=0.6,edgecolor='white')
axA.set_ylabel('Climate share (CV R²)'); axA.set_title('Climate share\nof yield var.'); axA.set_ylim(0,0.32)
for i,v in enumerate([a['yield_climate_share_CV_R2_RF'],a['yield_climate_share_CV_R2_linear']]): axA.text(i,v+0.008,f'{v:.2f}',ha='center',fontsize=6.3)
PL(axA,'a')
axB.bar(['Tech.','Clim.'],[a['yield_trend_tech_pct35'],a['yield_trend_climate_pct35']],color=[OI['orange'],OI['verm']],width=0.6,edgecolor='white')
axB.set_ylabel('Yield change 85–20 (%)'); axB.set_title('Attribution\nof yield trend')
for i,v in enumerate([a['yield_trend_tech_pct35'],a['yield_trend_climate_pct35']]): axB.text(i,v+0.5,f'{v:.1f}',ha='center',fontsize=6.3)
PL(axB,'b')
for yi,fi in enumerate(order):
    s=sv[:,fi]; xv=X[:,fi]; v=np.clip((xv-np.nanpercentile(xv,5))/(np.nanpercentile(xv,95)-np.nanpercentile(xv,5)+1e-9),0,1)
    j=(np.random.RandomState(fi).rand(len(s))-0.5)*0.34
    scp=axC.scatter(s,np.full_like(s,yi)+j,c=v,cmap='coolwarm',s=4,alpha=0.7,edgecolor='none',rasterized=True)
axC.axvline(0,color='0.6',lw=0.6); axC.set_yticks(range(len(order))); axC.set_yticklabels([NICE[CLIM[i]] for i in order])
axC.set_xlabel('SHAP value (→ higher yield)'); axC.set_title('Climate drivers of yield (SHAP)'); PL(axC,'c')
cb=fig.colorbar(scp,ax=axC,shrink=0.55,pad=0.008,ticks=[0,1],aspect=12); cb.set_label('Feature value',fontsize=6); cb.ax.set_yticklabels(['Lo','Hi'],fontsize=5.5)
axD.plot(px,py,'-',color=OI['verm'],lw=1.3); axD.axhline(0,color='0.7',lw=0.5)
axD.plot(X[:,topi],np.full(len(X),axD.get_ylim()[0]),'|',color='0.5',ms=3,alpha=0.3)
axD.set_xlabel(NICE[top]+' anom. (°C)'); axD.set_ylabel('Partial effect, ln(yield)',labelpad=1); axD.set_title('Heat penalty\n(nonlinear)'); PL(axD,'d')
fig.savefig(PUB+'/Fig8_attribution.pdf',bbox_inches='tight'); fig.savefig(PUB+'/Fig8_attribution.png',dpi=600,bbox_inches='tight'); fig.savefig(PUB+'/Fig8_attribution.svg',bbox_inches='tight'); plt.close()
print('Fig8 fixed (1x4), top=',top)
