import json,numpy as np,pandas as pd
import matplotlib as mpl; mpl.use('Agg'); import matplotlib.pyplot as plt
plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['DejaVu Sans'],'font.size':7,'axes.titlesize':8,
 'axes.labelsize':7,'xtick.labelsize':6.5,'ytick.labelsize':6.5,'axes.linewidth':0.6,'savefig.dpi':600,'pdf.fonttype':42})
OI={'blue':'#0072B2','orange':'#E69F00','green':'#009E73','verm':'#D55E00','sky':'#56B4E9','purple':'#CC79A7'}
MM=1/25.4; DC=183*MM
ROOT='/sessions/vibrant-bold-franklin/mnt/水稻'; MID=ROOT+'/增温驱动吉林水田扩张与增产_1985-2020/02_中间数据'; PUB=ROOT+'/增温驱动吉林水田扩张与增产_1985-2020/03_figures/pub'
a=json.load(open(MID+'/attribution_results.json')); z=np.load(MID+'/pdp_data.npz',allow_pickle=True)
NICE={'gs_tmean':'GS mean T','gdd':'GDD','at10':'AAT≥10°C','t_warm':'Warmest-month T','t_may':'May T','prec_gs':'GS precip','prec_78':'Jul–Aug precip','prec_conc':'Precip concentration'}
fig,ax=plt.subplots(1,4,figsize=(DC,DC*0.24))
# (a) climate share
ax[0].bar(['RF','Linear'],[a['yield_climate_share_CV_R2_RF'],a['yield_climate_share_CV_R2_linear']],color=[OI['green'],OI['sky']],width=0.6,edgecolor='white')
ax[0].set_ylabel('Climate share of yield\nvariability (CV R²)'); ax[0].set_title('Climate share'); ax[0].set_ylim(0,0.35)
for i,v in enumerate([a['yield_climate_share_CV_R2_RF'],a['yield_climate_share_CV_R2_linear']]): ax[0].text(i,v+0.01,f'{v:.2f}',ha='center',fontsize=6.5)
# (b) yield trend decomposition
ax[1].bar(['Technology','Climate'],[a['yield_trend_tech_pct35'],a['yield_trend_climate_pct35']],color=[OI['orange'],OI['verm']],width=0.6,edgecolor='white')
ax[1].set_ylabel('Yield change 1985–2020 (%)'); ax[1].set_title('Yield trend: tech vs climate')
for i,v in enumerate([a['yield_trend_tech_pct35'],a['yield_trend_climate_pct35']]): ax[1].text(i,v+0.6,f'{v:.1f}%',ha='center',fontsize=6.5)
# (c) SHAP importance yield
imp=pd.Series(a['shap_importance_yield']).sort_values()
ax[2].barh([NICE[k] for k in imp.index],imp.values,color=OI['green'],edgecolor='white')
ax[2].set_xlabel('mean |SHAP|'); ax[2].set_title('Climate drivers of yield')
# (d) PDP top factor
ax[3].plot(z['x'],z['y'],'-',color=OI['verm'],lw=1.3)
ax[3].set_xlabel(NICE.get(str(z['factor']),str(z['factor']))+' (°C)'); ax[3].set_ylabel('Partial effect on ln(yield)')
ax[3].set_title('Warmest-month heat penalty')
for a_ in ax: 
    for s in ['top','right']: a_.spines[s].set_visible(False)
for a_,L in zip(ax,'abcd'): a_.text(-0.18,1.08,L,transform=a_.transAxes,fontsize=9,fontweight='bold')
fig.tight_layout(); fig.savefig(PUB+'/Fig8_attribution.pdf',bbox_inches='tight'); fig.savefig(PUB+'/Fig8_attribution.png',bbox_inches='tight',dpi=600); plt.close()
# expansion SHAP
fig,ax=plt.subplots(figsize=(DC*0.42,DC*0.30)); impe=pd.Series(a['shap_importance_expansion']).sort_values()
ax.barh([NICE[k] for k in impe.index],impe.values,color=OI['blue'],edgecolor='white')
ax.set_xlabel('mean |SHAP|'); ax.set_title(f"Climate drivers of expansion (CV R²={a['expansion_climate_CV_R2']:.2f})")
for s in ['top','right']: ax.spines[s].set_visible(False)
fig.tight_layout(); fig.savefig(PUB+'/Fig9_expansion_drivers.pdf',bbox_inches='tight'); fig.savefig(PUB+'/Fig9_expansion_drivers.png',bbox_inches='tight',dpi=600); plt.close()
print('saved Fig8, Fig9')
