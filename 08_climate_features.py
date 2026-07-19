# -*- coding: utf-8 -*-
"""从逐月T/P派生县-年农业气候特征。"""
import numpy as np, pandas as pd
ROOT='/sessions/vibrant-bold-franklin/mnt/水稻'; MID=ROOT+'/增温驱动吉林水田扩张与增产_1985-2020/02_中间数据'
t=pd.read_csv(ROOT+'/climate_output/temp_monthly_detail.csv')
p=pd.read_csv(ROOT+'/climate_output/prec_monthly_detail.csv')
t['adcode']=t['adcode'].astype(str); p['adcode']=p['adcode'].astype(str)
DIM={1:31,2:28,3:31,4:30,5:31,6:30,7:31,8:31,9:30,10:31,11:30,12:31}
t['days']=t['month'].map(DIM)
# 生长季 5-9
gs=t[t.month.between(5,9)].copy()
gs['gdd']=np.clip(gs['temp_C']-10,0,None)*gs['days']       # base-10 GDD (approx)
gs['at10']=np.where(gs['temp_C']>=10,gs['temp_C']*gs['days'],0)  # 活动积温≥10
feat_t=gs.groupby(['adcode','year']).agg(
    gs_tmean=('temp_C','mean'), gdd=('gdd','sum'), at10=('at10','sum')).reset_index()
# 暖月(7-8)高温代理 & 5月冷凉代理
warm=t[t.month.isin([7,8])].groupby(['adcode','year'])['temp_C'].mean().rename('t_warm').reset_index()
may=t[t.month==5].groupby(['adcode','year'])['temp_C'].mean().rename('t_may').reset_index()
# 降水
pp=p.copy(); pgs=pp[pp.month.between(5,9)]
prec_gs=pgs.groupby(['adcode','year'])['prec_mm_total'].sum().rename('prec_gs').reset_index()
prec_78=pp[pp.month.isin([7,8])].groupby(['adcode','year'])['prec_mm_total'].sum().rename('prec_78').reset_index()
# 降水集中度(7-8占生长季)
F=feat_t.merge(warm,on=['adcode','year']).merge(may,on=['adcode','year']).merge(prec_gs,on=['adcode','year']).merge(prec_78,on=['adcode','year'])
F['prec_conc']=F['prec_78']/F['prec_gs'].replace(0,np.nan)
F=F.rename(columns={'adcode':'CountyCode'})
F.to_csv(MID+'/climate_features_county_year.csv',index=False,encoding='utf-8-sig')
print('features rows',len(F),'cols',list(F.columns))
print(F.describe().round(1).to_string())
