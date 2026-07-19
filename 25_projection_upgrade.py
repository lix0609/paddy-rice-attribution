# -*- coding: utf-8 -*-
"""A1-A5 预测升级：RF预测内核(报告CV技巧) + 绝对产量(吨) + 净产量对冲 + 分带赢家输家。"""
import numpy as np, pandas as pd, json, warnings; warnings.filterwarnings('ignore')
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.metrics import r2_score
import statsmodels.api as sm
PROJ='/sessions/vibrant-bold-franklin/mnt/水稻/增温驱动吉林水田扩张与增产_1985-2020'; MID=PROJ+'/02_中间数据'
DIM={5:31,6:30,7:31,8:31,9:30}
CLIM=['gs_tmean','gdd','at10','t_warm','t_may','prec_gs','prec_78','prec_conc']

def feats(tas,pr):
    gs=[5,6,7,8,9]
    gs_tmean=np.mean([tas[m] for m in gs]); t_warm=np.mean([tas[7],tas[8]]); t_may=tas[5]
    gdd=sum(max(tas[m]-10,0)*DIM[m] for m in gs)
    at10=sum(tas[m]*DIM[m] for m in gs if tas[m]>=10)
    prec_gs=sum(pr[m] for m in gs); prec_78=pr[7]+pr[8]; prec_conc=prec_78/prec_gs if prec_gs else np.nan
    return dict(gs_tmean=gs_tmean,gdd=gdd,at10=at10,t_warm=t_warm,t_may=t_may,prec_gs=prec_gs,prec_78=prec_78,prec_conc=prec_conc)

# ---------- A1: RF 归因/预测内核 ----------
pan=pd.read_csv(MID+'/county_year_panel.csv'); pan['CountyCode']=pan['CountyCode'].astype(str)
F=pd.read_csv(MID+'/climate_features_county_year.csv'); F['CountyCode']=F['CountyCode'].astype(str)
df=pan.merge(F,on=['CountyCode','year'],how='left').dropna(subset=['yield_kg_ha']+CLIM).copy()
df=df[df.gdd>0]; df['ln_yield']=np.log(df['yield_kg_ha'])
def detrend(g):
    if g['year'].nunique()<8: g['clim_resid']=np.nan; return g
    b=np.polyfit(g['year'],g['ln_yield'],1); g['clim_resid']=g['ln_yield']-np.polyval(b,g['year']); return g
df=df.groupby('CountyCode',group_keys=False).apply(detrend).dropna(subset=['clim_resid'])
for c in CLIM: df[c+'_anom']=df[c]-df.groupby('CountyCode')[c].transform('mean')
ANOM=[c+'_anom' for c in CLIM]
X=df[ANOM].values; y=df['clim_resid'].values; grp=df['CountyCode'].values
gkf=GroupKFold(n_splits=5)
rf=RandomForestRegressor(n_estimators=400,min_samples_leaf=5,n_jobs=-1,random_state=42)
yhat_cv=cross_val_predict(rf,X,y,cv=gkf,groups=grp,n_jobs=-1)
cvR2=r2_score(y,yhat_cv)
# 观测-预测斜率
slope_op=np.polyfit(yhat_cv,y,1)[0]
rf.fit(X,y)
# 历史暖月异常范围(用于外推说明)
tw_anom_hist=(df['t_warm']-df.groupby('CountyCode')['t_warm'].transform('mean'))
print('A1 RF: CV R2=%.3f  obs-pred slope=%.2f  |t_warm anomaly hist [%.1f,%.1f]'%(cvR2,slope_op,tw_anom_hist.min(),tw_anom_hist.max()))

# ---------- 二次响应模型(灵敏度对照) ----------
FF=df.copy()
Xq=sm.add_constant(pd.DataFrame({'gs':FF.gs_tmean,'tw':FF.t_warm,'tw2':FF.t_warm**2,'pr':FF.prec_gs}))
modq=sm.OLS(FF.clim_resid,Xq).fit()
def predq(gs,tw,pr):
    return modq.predict(sm.add_constant(pd.DataFrame({'gs':gs,'tw':tw,'tw2':tw**2,'pr':pr}),has_constant='add'))

# ---------- CMIP6 delta ----------
cm=pd.read_csv(MID+'/cmip6_county_monthly.csv'); cm['CountyCode']=cm['CountyCode'].astype(str)
def get(ds):
    d=cm[cm.dataset==ds].set_index('CountyCode'); return {c:{m:d.loc[c,'m%02d'%m] for m in range(1,13)} for c in d.index}
tasB,prB=get('tas_base'),get('pr_base'); tas245,pr245=get('tas_245'),get('pr_245'); tas585,pr585=get('tas_585'),get('pr_585')
codes=[c for c in tasB if c in prB and c in tas245]
BASE=pd.DataFrame([feats(tasB[c],prB[c]) for c in codes],index=codes)[CLIM]
F245=pd.DataFrame([feats(tas245[c],pr245[c]) for c in codes],index=codes)[CLIM]
F585=pd.DataFrame([feats(tas585[c],pr585[c]) for c in codes],index=codes)[CLIM]
d245=(F245-BASE); d585=(F585-BASE)
# 观测基准(1991-2020县均)
obsB=F[(F.year>=1991)&(F.year<=2020)&(F.gdd>0)].groupby('CountyCode')[CLIM].mean()
idx=[c for c in obsB.index if c in d245.index]
base=obsB.loc[idx]; p245=base+d245.loc[idx]; p585=base+d585.loc[idx]

def dY_rf(delta):
    z=np.zeros((len(delta),len(CLIM)))
    return 100*(rf.predict(delta[CLIM].values)-rf.predict(z))
dy245_rf=dY_rf(d245.loc[idx]); dy585_rf=dY_rf(d585.loc[idx])
# 二次灵敏度
dy245_q=100*(predq(p245.gs_tmean,p245.t_warm,p245.prec_gs).values-predq(base.gs_tmean,base.t_warm,base.prec_gs).values)
dy585_q=100*(predq(p585.gs_tmean,p585.t_warm,p585.prec_gs).values-predq(base.gs_tmean,base.t_warm,base.prec_gs).values)
print('A1 ΔY省均: RF 245=%.2f%% 585=%.2f%% | 二次 245=%.2f%% 585=%.2f%%'%(dy245_rf.mean(),dy585_rf.mean(),dy245_q.mean(),dy585_q.mean()))

# ---------- 热量适宜 ----------
REQ=2840
suit_base=100*(base.at10>=REQ).mean(); suit245=100*(p245.at10>=REQ).mean(); suit585=100*(p585.at10>=REQ).mean()

# 预测主用【可外推的多变量二次响应】; RF 因未来超出历史包络而饱和, 仅作技巧与符号交叉验证
dy245=dy245_q; dy585=dy585_q
print('  [说明] 未来暖月温超出历史包络, RF饱和(见上); 逐县预测采用可外推的多变量二次响应')

# ---------- A2 绝对产量影响(吨) ----------
recent=pan[(pan.year>=2016)&(pan.year<=2020)].groupby('CountyCode').agg(prod_kg=('prod_kg','mean'),paddy_ha=('paddy_ha','mean'),yield_kg_ha=('yield_kg_ha','mean')).reindex(idx)
prod_t=recent['prod_kg']/1000.0   # tonnes
prov_prod_t=prod_t.sum()
loss245_t=float((prod_t.values*dy245/100).sum()); loss585_t=float((prod_t.values*dy585/100).sum())
print('A2 存量面积气候单产影响: 245=%.0f t (%.2f%%) / 585=%.0f t (%.2f%%) | 省近期总产%.0f t'%(
  loss245_t,100*loss245_t/prov_prod_t,loss585_t,100*loss585_t/prov_prod_t,prov_prod_t))

# ---------- A3 净产量对冲(扩张增产 vs 单产热惩罚) ----------
BETA=0.23379845441467773  # 面积~生长季温 弹性 (regression_results, p=0.042)
dT245=float((p245.gs_tmean-base.gs_tmean).mean()); dT585=float((p585.gs_tmean-base.gs_tmean).mean())
A2020=recent['paddy_ha'].sum(); ybar=recent['yield_kg_ha'].mean()  # kg/ha
def expansion_gain_t(dT):
    dArea=A2020*BETA*dT                # ha 潜在新增(气候可归因)
    return dArea*ybar/1000.0, dArea    # tonnes
gain245_t,dA245=expansion_gain_t(dT245); gain585_t,dA585=expansion_gain_t(dT585)
net245=gain245_t+loss245_t; net585=gain585_t+loss585_t
print('A3 净产量对冲:')
print('  245: 扩张潜在+%.0f t (新增%.0f ha) 单产%.0f t 净=%+.0f t'%(gain245_t,dA245,loss245_t,net245))
print('  585: 扩张潜在+%.0f t (新增%.0f ha) 单产%.0f t 净=%+.0f t'%(gain585_t,dA585,loss585_t,net585))

# ---------- A4 分纬度带 ----------
st=pan.groupby('CountyCode')['lat'].mean().reindex(idx)
q1,q2=st.quantile([1/3,2/3])
band=pd.Series(np.where(st<=q1,'South',np.where(st<=q2,'Central','North')),index=idx)
res=pd.DataFrame({'CountyCode':idx,'Name':pan.groupby('CountyCode')['Name'].first().reindex(idx).values,
  'lat':st.values,'band':band.values,'dY245':dy245,'dY585':dy585,
  'dY245_rf':dy245_rf,'dY585_rf':dy585_rf,
  't_warm_245':p245.t_warm.values,'t_warm_585':p585.t_warm.values,
  'at10_base':base.at10.values,'at10_245':p245.at10.values,'at10_585':p585.at10.values,
  'prod_recent_t':prod_t.values})
bands={}
for b in ['North','Central','South']:
    g=res[res.band==b]
    bands[b]={'n':int(len(g)),'dY245_mean':float(g.dY245.mean()),'dY585_mean':float(g.dY585.mean()),
      'n_loss245':int((g.dY245<0).sum()),'n_loss585':int((g.dY585<0).sum()),
      'dY585_min':float(g.dY585.min()),'dY585_max':float(g.dY585.max()),
      'twarm585_mean':float(g.t_warm_585.mean())}
print('A4 分带 ΔY(585)/减产县:',{b:(round(bands[b]['dY585_mean'],2),bands[b]['n_loss585'],bands[b]['n']) for b in bands})

# ---------- A5 赢家/输家 ----------
def stats(dy):
    dy=pd.Series(dy); return dict(n=int(len(dy)),n_loss=int((dy<0).sum()),pct_loss=float(100*(dy<0).mean()),
      mean=float(dy.mean()),median=float(dy.median()),worst=float(dy.min()),best=float(dy.max()))
s245=stats(dy245); s585=stats(dy585)
worst5=res.nsmallest(5,'dY585')[['Name','band','dY585','t_warm_585']]
print('A5 585:',s585); print('  暴露最重5县:'); print(worst5.to_string(index=False))

res.to_csv(MID+'/projection_county_v2.csv',index=False,encoding='utf-8-sig')
out={}
out['model']={'cv_R2_rf':float(cvR2),'obs_pred_slope':float(slope_op),'quad_R2':float(modq.rsquared),
    'twarm_anom_hist_min':float(tw_anom_hist.min()),'twarm_anom_hist_max':float(tw_anom_hist.max())}
out['warming']={'dT_gs_245':dT245,'dT_gs_585':dT585,'dAT10_245':float((p245.at10-base.at10).mean()),'dAT10_585':float((p585.at10-base.at10).mean())}
out['suitability_pct']={'base':float(suit_base),'ssp245':float(suit245),'ssp585':float(suit585)}
out['dY_primary_quad']={'ssp245_mean':float(dy245.mean()),'ssp585_mean':float(dy585.mean())}
out['dY_rf_crosscheck_saturated']={'ssp245_mean':float(dy245_rf.mean()),'ssp585_mean':float(dy585_rf.mean())}
out['production']={'prov_recent_t':float(prov_prod_t),'yield_loss_t':{'ssp245':loss245_t,'ssp585':loss585_t},
    'yield_loss_pct':{'ssp245':100*loss245_t/prov_prod_t,'ssp585':100*loss585_t/prov_prod_t}}
out['net_outlook_t']={'ssp245':{'expansion_gain':gain245_t,'new_area_ha':float(dA245),'yield_effect':loss245_t,'net':net245},
    'ssp585':{'expansion_gain':gain585_t,'new_area_ha':float(dA585),'yield_effect':loss585_t,'net':net585}}
out['bands']=bands
out['winners_losers']={'ssp245':s245,'ssp585':s585}
w5=worst5.copy()
w5['dY585']=w5['dY585'].round(2)
w5['t_warm_585']=w5['t_warm_585'].round(2)
out['most_exposed']=w5.to_dict('records')
json.dump(out,open(MID+'/projection_upgrade.json','w'),ensure_ascii=False,indent=2,default=float)
print('\nsaved projection_county_v2.csv & projection_upgrade.json')
