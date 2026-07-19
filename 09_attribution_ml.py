# -*- coding: utf-8 -*-
import os, json, warnings
import numpy as np, pandas as pd
warnings.filterwarnings('ignore')
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.inspection import partial_dependence
from sklearn.metrics import r2_score
import shap
ROOT='/sessions/vibrant-bold-franklin/mnt/水稻'; MID=ROOT+'/增温驱动吉林水田扩张与增产_1985-2020/02_中间数据'
pan=pd.read_csv(MID+'/county_year_panel.csv'); pan['CountyCode']=pan['CountyCode'].astype(str)
F=pd.read_csv(MID+'/climate_features_county_year.csv'); F['CountyCode']=F['CountyCode'].astype(str)
CLIM=['gs_tmean','gdd','at10','t_warm','t_may','prec_gs','prec_78','prec_conc']
df=pan.merge(F,on=['CountyCode','year'],how='left').dropna(subset=['yield_kg_ha']+CLIM).copy()
df=df[df.gdd>0]; df['ln_yield']=np.log(df['yield_kg_ha'])
def detrend(g):
    if g['year'].nunique()<8: g['tech']=np.nan; g['clim_resid']=np.nan; return g
    b=np.polyfit(g['year'],g['ln_yield'],1); g['tech']=np.polyval(b,g['year']); g['clim_resid']=g['ln_yield']-g['tech']; return g
df=df.groupby('CountyCode',group_keys=False).apply(detrend).dropna(subset=['clim_resid'])
for c in CLIM: df[c+'_anom']=df[c]-df.groupby('CountyCode')[c].transform('mean')
ANOM=[c+'_anom' for c in CLIM]
X=df[ANOM].values; y=df['clim_resid'].values; grp=df['CountyCode'].values
gkf=GroupKFold(n_splits=5)
rf=RandomForestRegressor(n_estimators=300,min_samples_leaf=5,n_jobs=-1,random_state=42)
r2_rf=r2_score(y,cross_val_predict(rf,X,y,cv=gkf,groups=grp,n_jobs=-1))
r2_lin=r2_score(y,cross_val_predict(LinearRegression(),X,y,cv=gkf,groups=grp))
var_share=float(np.var(df['clim_resid'])/np.var(df['ln_yield']-df['ln_yield'].mean()))
print('气候解释产量年际波动 CV R2: RF=%.3f Linear=%.3f'%(r2_rf,r2_lin))
rf.fit(X,y); sv=shap.TreeExplainer(rf).shap_values(X)
imp=pd.Series(np.abs(sv).mean(0),index=CLIM).sort_values(ascending=False)
print('SHAP(气候->产量残差):'); print(imp.round(4).to_string())
df['clim_pred']=rf.predict(X)
prov=df.groupby('year').agg(ln_tech=('tech','mean'),ln_climpred=('clim_pred','mean')).reset_index()
sl=lambda a,b:np.polyfit(a,b,1)[0]
tech_tr=sl(prov.year,prov.ln_tech)*35*100; clim_tr=sl(prov.year,prov.ln_climpred)*35*100
print('单产对数趋势(35年,%%): 技术=%.1f 气候=%.1f'%(tech_tr,clim_tr))
top=imp.index[0]; pdp=partial_dependence(rf,X,[CLIM.index(top)],grid_resolution=40)
pdp_x=np.array(pdp['grid_values'][0]); pdp_y=np.array(pdp['average'][0])
# 扩张
de=pan.merge(F,on=['CountyCode','year'],how='left').dropna(subset=['paddy_ha']+CLIM); de=de[(de.paddy_ha>0)&(de.gdd>0)].copy()
de['ln_paddy']=np.log(de['paddy_ha'])
for c in CLIM: de[c+'_anom']=de[c]-de.groupby('CountyCode')[c].transform('mean')
de['ln_paddy_dm']=de['ln_paddy']-de.groupby('CountyCode')['ln_paddy'].transform('mean')
Xe=de[ANOM].values; ye=de['ln_paddy_dm'].values; grpe=de['CountyCode'].values
rfe=RandomForestRegressor(n_estimators=300,min_samples_leaf=5,n_jobs=-1,random_state=42)
r2e=r2_score(ye,cross_val_predict(rfe,Xe,ye,cv=gkf,groups=grpe,n_jobs=-1))
rfe.fit(Xe,ye); impe=pd.Series(np.abs(shap.TreeExplainer(rfe).shap_values(Xe)).mean(0),index=CLIM).sort_values(ascending=False)
print('气候解释扩张 CV R2=%.3f'%r2e); print(impe.round(4).to_string())
out={'yield_climate_share_CV_R2_RF':float(r2_rf),'yield_climate_share_CV_R2_linear':float(r2_lin),
 'clim_component_var_share':var_share,'yield_trend_tech_pct35':float(tech_tr),'yield_trend_climate_pct35':float(clim_tr),
 'shap_importance_yield':imp.round(5).to_dict(),'expansion_climate_CV_R2':float(r2e),
 'shap_importance_expansion':impe.round(5).to_dict(),'pdp_top_factor':top,'n_yield':int(len(df)),'n_exp':int(len(de))}
json.dump(out,open(MID+'/attribution_results.json','w'),ensure_ascii=False,indent=2,default=float)
np.savez(MID+'/pdp_data.npz',x=pdp_x,y=pdp_y,factor=top,shap_yn=list(imp.index),shap_yv=imp.values,shap_en=list(impe.index),shap_ev=impe.values)
print('saved.')
