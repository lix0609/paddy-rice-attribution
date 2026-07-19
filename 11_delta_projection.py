# -*- coding: utf-8 -*-
"""Delta-change 未来预估(线性气候敏感性,文献CMIP6增量)。透明、可外推、带警示。"""
import json,numpy as np,pandas as pd,geopandas as gpd,warnings; warnings.filterwarnings('ignore')
from linearmodels.panel import PanelOLS
import matplotlib as mpl; mpl.use('Agg'); import matplotlib.pyplot as plt
plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['DejaVu Sans'],'font.size':7,'axes.titlesize':8,'axes.labelsize':7,'xtick.labelsize':6.5,'ytick.labelsize':6.5,'axes.linewidth':0.6,'savefig.dpi':600,'pdf.fonttype':42})
OI={'blue':'#0072B2','orange':'#E69F00','green':'#009E73','verm':'#D55E00'}; MM=1/25.4; DC=183*MM
ROOT='/sessions/vibrant-bold-franklin/mnt/水稻'; MID=ROOT+'/增温驱动吉林水田扩张与增产_1985-2020/02_中间数据'; PUB=ROOT+'/增温驱动吉林水田扩张与增产_1985-2020/03_figures/pub'
DELTA={'SSP2-4.5':1.8,'SSP5-8.5':2.5}  # NE China 生长季 mid-century 增温(文献central,°C)
pan=pd.read_csv(MID+'/county_year_panel.csv'); pan['CountyCode']=pan['CountyCode'].astype(str)
F=pd.read_csv(MID+'/climate_features_county_year.csv'); F['CountyCode']=F['CountyCode'].astype(str)
df=pan.merge(F,on=['CountyCode','year'],how='left')

# --- 单产气候分量线性敏感性(去趋势残差 ~ 关键温度), 含热量惩罚 ---
d=df.dropna(subset=['yield_kg_ha','gs_tmean','t_warm','prec_gs']).copy(); d['ln_yield']=np.log(d['yield_kg_ha'])
def dt(g):
    if g.year.nunique()<8: g['r']=np.nan;return g
    g['r']=g['ln_yield']-np.polyval(np.polyfit(g.year,g.ln_yield,1),g.year); return g
d=d.groupby('CountyCode',group_keys=False).apply(dt).dropna(subset=['r'])
for c in ['gs_tmean','t_warm','prec_gs']: d[c+'_a']=d[c]-d.groupby('CountyCode')[c].transform('mean')
import statsmodels.api as sm
Xr=sm.add_constant(d[['gs_tmean_a','t_warm_a','prec_gs_a']]); m=sm.OLS(d['r'],Xr).fit()
b_gs=m.params['gs_tmean_a']; b_warm=m.params['t_warm_a']
# 温升同时抬高 gs_tmean 与 t_warm → 单产气候分量敏感性(每°C)
yield_sens=b_gs+b_warm
print('单产气候分量温度敏感性: gs=%.4f warm=%.4f 合计=%.4f /°C (p_gs=%.3f p_warm=%.3f)'%(b_gs,b_warm,yield_sens,m.pvalues['gs_tmean_a'],m.pvalues['t_warm_a']))

# --- 扩张热量敏感性(县FE, 线性, 可外推) ---
de=df.dropna(subset=['paddy_ha','gs_tmean']); de=de[de.paddy_ha>0].copy(); de['ln_paddy']=np.log(de['paddy_ha'])
di=de.set_index(['CountyCode','year']); rr=PanelOLS(di['ln_paddy'],di[['gs_tmean']],entity_effects=True,check_rank=False).fit(cov_type='clustered',cluster_entity=True)
exp_sens=rr.params['gs_tmean']; print('扩张热量敏感性: %.3f /°C (p=%.3f)'%(exp_sens,rr.pvalues['gs_tmean']))

# --- 分县基线(近10年)与投影 ---
base=df[df.year>=2011].groupby('CountyCode').agg(gs_tmean=('gs_tmean','mean'),lat=('lat','mean')).reset_index()
st=pd.read_csv(MID+'/county_static.csv'); st['CountyCode']=st['CountyCode'].astype(str)
q=pd.qcut(base.set_index('CountyCode')['lat'].dropna(),3,labels=['South','Central','North'])
base['band']=base['CountyCode'].map(q)
res={'delta_C':DELTA,'yield_climate_sens_per_C':float(yield_sens),'expansion_sens_per_C':float(exp_sens)}
rows=[]
for ssp,dT in DELTA.items():
    base['dY_climate_pct']=100*(np.exp(yield_sens*dT)-1)
    base['dExp_pct']=100*(np.exp(exp_sens*dT)-1)
    for band,g in base.groupby('band'):
        rows.append((ssp,band,g['dY_climate_pct'].mean(),g['dExp_pct'].mean()))
    res[ssp]={'yield_climate_change_pct':float(100*(np.exp(yield_sens*dT)-1)),
              'expansion_potential_pct':float(100*(np.exp(exp_sens*dT)-1))}
proj=pd.DataFrame(rows,columns=['SSP','band','dY_climate_pct','dExp_pct'])
proj.to_csv(MID+'/projection_bands.csv',index=False); json.dump(res,open(MID+'/projection_results.json','w'),indent=2,default=float)
print(proj.round(1).to_string(index=False)); print('\n省级:',{k:res[k] for k in DELTA})

# --- 图: (a)单产气候分量变化 (b)扩张潜力, 按SSP与纬度带 ---
fig,ax=plt.subplots(1,2,figsize=(DC*0.7,DC*0.30)); bands=['South','Central','North']; x=np.arange(3); w=0.35
for i,ssp in enumerate(DELTA):
    sub=proj[proj.SSP==ssp].set_index('band').loc[bands]
    ax[0].bar(x+(i-0.5)*w,sub['dY_climate_pct'],w,label=ssp,color=[OI['orange'],OI['verm']][i],edgecolor='white')
    ax[1].bar(x+(i-0.5)*w,sub['dExp_pct'],w,label=ssp,color=[OI['blue'],OI['green']][i],edgecolor='white')
ax[0].set_ylabel('Climate-driven yield change (%)'); ax[0].set_title('(a) Yield (climate component)'); ax[0].axhline(0,color='k',lw=0.5)
ax[1].set_ylabel('Thermal expansion potential (%)'); ax[1].set_title('(b) Expansion potential')
for a in ax: a.set_xticks(x); a.set_xticklabels(bands); a.legend(frameon=False,fontsize=6); 
for a in ax:
    for s in ['top','right']: a.spines[s].set_visible(False)
for a,L in zip(ax,'ab'): a.text(-0.16,1.06,L,transform=a.transAxes,fontsize=9,fontweight='bold')
fig.tight_layout(); fig.savefig(PUB+'/Fig10_projection.pdf',bbox_inches='tight'); fig.savefig(PUB+'/Fig10_projection.png',bbox_inches='tight',dpi=600); plt.close()
print('saved Fig10 + projection results')
