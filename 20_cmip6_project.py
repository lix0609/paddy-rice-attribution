# -*- coding: utf-8 -*-
"""Delta-change 未来预测: 单产气候变化 + 热量适宜扩张潜力 (SSP245/585, 2041-2070)。"""
import numpy as np,pandas as pd,json,warnings; warnings.filterwarnings('ignore')
import statsmodels.api as sm
PROJ='/sessions/vibrant-bold-franklin/mnt/水稻/增温驱动吉林水田扩张与增产_1985-2020'; MID=PROJ+'/02_中间数据'
DIM={5:31,6:30,7:31,8:31,9:30}
def feats(tas,pr):
    # tas,pr: dict month->value (1..12)
    gs=[5,6,7,8,9]
    gs_tmean=np.mean([tas[m] for m in gs]); t_warm=np.mean([tas[7],tas[8]]); t_may=tas[5]
    gdd=sum(max(tas[m]-10,0)*DIM[m] for m in gs)
    at10=sum(tas[m]*DIM[m] for m in gs if tas[m]>=10)
    prec_gs=sum(pr[m] for m in gs); prec_78=pr[7]+pr[8]; prec_conc=prec_78/prec_gs if prec_gs else np.nan
    return dict(gs_tmean=gs_tmean,t_warm=t_warm,t_may=t_may,gdd=gdd,at10=at10,prec_gs=prec_gs,prec_78=prec_78,prec_conc=prec_conc)
# CMIP6 分县月度
cm=pd.read_csv(MID+'/cmip6_county_monthly.csv'); cm['CountyCode']=cm['CountyCode'].astype(str)
def get(ds):
    d=cm[cm.dataset==ds].set_index('CountyCode')
    return {c:{m:d.loc[c,'m%02d'%m] for m in range(1,13)} for c in d.index}
tasB,prB=get('tas_base'),get('pr_base'); tas245,pr245=get('tas_245'),get('pr_245'); tas585,pr585=get('tas_585'),get('pr_585')
codes=[c for c in tasB if c in prB]
CF=[feats(tasB[c],prB[c]) for c in codes]
rowsB=pd.DataFrame(CF,index=codes)
F245=pd.DataFrame([feats(tas245[c],pr245[c]) for c in codes],index=codes)
F585=pd.DataFrame([feats(tas585[c],pr585[c]) for c in codes],index=codes)
# CMIP6 delta (future - base)
d245=F245-rowsB; d585=F585-rowsB
# 观测基准(1991-2020县均)
obs=pd.read_csv(MID+'/climate_features_county_year.csv'); obs['CountyCode']=obs['CountyCode'].astype(str)
obsB=obs[(obs.year>=1991)&(obs.year<=2020)&(obs.gdd>0)].groupby('CountyCode')[['gs_tmean','t_warm','t_may','gdd','at10','prec_gs','prec_78','prec_conc']].mean()
FEAT=['gs_tmean','t_warm','t_may','gdd','at10','prec_gs','prec_78','prec_conc']
def project(delta):
    idx=[c for c in obsB.index if c in delta.index]
    return (obsB.loc[idx,FEAT]+delta.loc[idx,FEAT]).reindex(columns=FEAT), idx
proj245,idx=project(d245); proj585,_=project(d585)
baseObs=obsB.loc[idx,FEAT]
# 省均增温
print('省均: 生长季均温 基准%.1f -> SSP245 %.1f(+%.1f) / SSP585 %.1f(+%.1f)'%(
  baseObs.gs_tmean.mean(),proj245.gs_tmean.mean(),proj245.gs_tmean.mean()-baseObs.gs_tmean.mean(),
  proj585.gs_tmean.mean(),proj585.gs_tmean.mean()-baseObs.gs_tmean.mean()))
print('省均: 暖月温 基准%.1f -> 245 %.1f / 585 %.1f'%(baseObs.t_warm.mean(),proj245.t_warm.mean(),proj585.t_warm.mean()))
print('省均: 积温≥10 基准%.0f -> 245 %.0f(+%.0f) / 585 %.0f(+%.0f)'%(
  baseObs.at10.mean(),proj245.at10.mean(),proj245.at10.mean()-baseObs.at10.mean(),
  proj585.at10.mean(),proj585.at10.mean()-baseObs.at10.mean()))

# ===== 单产响应模型: 去趋势残差 ~ gs_tmean + t_warm + t_warm^2 + prec_gs (吸收GS均温红利+暖月热惩罚) =====
pan=pd.read_csv(MID+'/county_year_panel.csv'); pan['CountyCode']=pan['CountyCode'].astype(str)
FF=obs.merge(pan[['CountyCode','year','yield_kg_ha']],on=['CountyCode','year'],how='left').dropna(subset=['yield_kg_ha','gs_tmean','t_warm','prec_gs'])
FF=FF[FF.gdd>0].copy(); FF['ln_y']=np.log(FF['yield_kg_ha'])
FF=FF.groupby('CountyCode',group_keys=False).apply(lambda g: g.assign(r=(g['ln_y']-np.polyval(np.polyfit(g.year,g.ln_y,1),g.year)) if g.year.nunique()>=8 else np.nan)).dropna(subset=['r'])
X=sm.add_constant(pd.DataFrame({'gs':FF.gs_tmean,'tw':FF.t_warm,'tw2':FF.t_warm**2,'pr':FF.prec_gs}))
mod=sm.OLS(FF.r,X).fit()
print('\n单产响应模型系数:',{k:round(v,4) for k,v in mod.params.items()},'  R2=%.3f'%mod.rsquared)
def predr(df):
    Xp=sm.add_constant(pd.DataFrame({'gs':df.gs_tmean,'tw':df.t_warm,'tw2':df.t_warm**2,'pr':df.prec_gs}),has_constant='add')
    return mod.predict(Xp)
dy245=100*(predr(proj245).values-predr(baseObs).values)
dy585=100*(predr(proj585).values-predr(baseObs).values)
res=pd.DataFrame({'CountyCode':idx,'dY_245':dy245,'dY_585':dy585,
                  'dAT10_245':(proj245.at10-baseObs.at10).values,'dAT10_585':(proj585.at10-baseObs.at10).values,
                  'at10_base':baseObs.at10.values,'at10_245':proj245.at10.values,'at10_585':proj585.at10.values,
                  't_warm_245':proj245.t_warm.values,'t_warm_585':proj585.t_warm.values})
res.to_csv(MID+'/projection_county.csv',index=False,encoding='utf-8-sig')
print('\n=== 单产气候变化(省均) ===')
print(' SSP245: %.1f%% (范围 %.1f ~ %.1f)'%(dy245.mean(),dy245.min(),dy245.max()))
print(' SSP585: %.1f%% (范围 %.1f ~ %.1f)'%(dy585.mean(),dy585.min(),dy585.max()))
print(' 输家(单产降)县数 245=%d/%d, 585=%d/%d'%((dy245<0).sum(),len(dy245),(dy585<0).sum(),len(dy585)))
# 热量适宜扩张: 达到成熟所需积温~2840的县比例(基准 vs 未来)
REQ=2840
print('\n=== 热量适宜(积温≥%d °C·d)县比例 ==='%REQ)
print(' 基准 %.0f%%, SSP245 %.0f%%, SSP585 %.0f%%'%(100*(baseObs.at10>=REQ).mean(),100*(proj245.at10>=REQ).mean(),100*(proj585.at10>=REQ).mean()))
json.dump({'gs_warm_245':float(proj245.gs_tmean.mean()-baseObs.gs_tmean.mean()),
           'gs_warm_585':float(proj585.gs_tmean.mean()-baseObs.gs_tmean.mean()),
           'dAT10_245':float((proj245.at10-baseObs.at10).mean()),'dAT10_585':float((proj585.at10-baseObs.at10).mean()),
           'dY_245_mean':float(dy245.mean()),'dY_585_mean':float(dy585.mean()),
           'model_params':{k:float(v) for k,v in mod.params.items()},'model_R2':float(mod.rsquared)},
          open(MID+'/projection_summary.json','w'),ensure_ascii=False,indent=2)
print('saved projection_county.csv & projection_summary.json')
