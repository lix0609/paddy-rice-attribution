# -*- coding: utf-8 -*-
import os, glob, re, json, warnings
import numpy as np, pandas as pd
warnings.filterwarnings('ignore')
from linearmodels.panel import PanelOLS
ROOT='/sessions/vibrant-bold-franklin/mnt/水稻'
MID=ROOT+'/增温驱动吉林水田扩张与增产_1985-2020/02_中间数据'

# --- paddy ---
pa=pd.read_csv(MID+'/paddy_area_by_county_1985_2020.csv'); pa['CountyCode']=pa['CountyCode'].astype(str)
# --- yield ---
rows=[]
for f in glob.glob(ROOT+'/产量数据/整理后的产量数据/Jilin_Rice_Panel_Data_*.csv'):
    m=re.search(r'(\d{4})',os.path.basename(f)); 
    if not m: continue
    y=int(m.group(1)); 
    if not (1985<=y<=2020): continue
    try: d=pd.read_csv(f,encoding='utf-8-sig')
    except: d=pd.read_csv(f,encoding='gbk')
    d.columns=[c.strip() for c in d.columns]
    code=d.columns[0]; ac=[c for c in d.columns if '面积' in c][0]; pc=[c for c in d.columns if '产量' in c][0]
    yc=[c for c in d.columns if '单产' in c]
    t=pd.DataFrame({'CountyCode':d[code].astype(str),'year':y,
                    'area_ha':pd.to_numeric(d[ac],errors='coerce'),
                    'prod_kg':pd.to_numeric(d[pc],errors='coerce')})
    rows.append(t)
yld=pd.concat(rows,ignore_index=True)
yld['yield_kg_ha']=yld['prod_kg']/yld['area_ha']
yld.loc[(yld.yield_kg_ha<3000)|(yld.yield_kg_ha>13000),['yield_kg_ha']]=np.nan
# --- climate long ---
def melt_clim(path,name):
    d=pd.read_excel(path); d['adcode']=d['adcode'].astype(str)
    yrs=[c for c in d.columns if str(c).isdigit()]
    m=d.melt(id_vars=['adcode'],value_vars=yrs,var_name='year',value_name=name)
    m['year']=m['year'].astype(int); m=m.rename(columns={'adcode':'CountyCode'})
    return m
ct=melt_clim(ROOT+'/climate_output/growing_temp_1985_2020.xlsx','gs_temp')
cp=melt_clim(ROOT+'/climate_output/growing_prec_1985_2020.xlsx','gs_prec')
st=pd.read_csv(MID+'/county_static.csv'); st['CountyCode']=st['CountyCode'].astype(str)

p=pa.merge(yld,on=['CountyCode','year'],how='outer').merge(ct,on=['CountyCode','year'],how='left')\
    .merge(cp,on=['CountyCode','year'],how='left').merge(st[['CountyCode','lat','lon','elev_mean']],on='CountyCode',how='left')
p=p.dropna(subset=['gs_temp'])
p['ln_paddy']=np.log(p['paddy_ha'].where(p['paddy_ha']>0))
p['ln_yield']=np.log(p['yield_kg_ha'])
p.to_csv(MID+'/county_year_panel.csv',index=False,encoding='utf-8-sig')
print('panel rows',len(p),'counties',p.CountyCode.nunique(),'years',p.year.min(),p.year.max())

res={}
def run(dfn,y,xs,label):
    d=dfn.dropna(subset=[y]+xs).copy()
    d=d.set_index(['CountyCode','year'])
    mod=PanelOLS(d[y],d[xs],entity_effects=True,check_rank=False)
    r=mod.fit(cov_type='clustered',cluster_entity=True)
    out={'n':int(r.nobs)}
    for x in xs: out[x]={'coef':float(r.params[x]),'p':float(r.pvalues[x]),'t':float(r.tstats[x])}
    print('\n===',label,'===\n',r.params.to_string(),'\n p:',{x:round(r.pvalues[x],4) for x in xs},'n=',int(r.nobs))
    return out,r

# 1. 扩张对增温响应
res['expansion'],_=run(p,'ln_paddy',['gs_temp','gs_prec'],'Expansion ~ temp+prec (county FE)')
# 2. 单产热量阈值(二次)
p['temp2']=p['gs_temp']**2
res['yield_quad'],ry=run(p,'ln_yield',['gs_temp','temp2','gs_prec'],'Yield ~ temp+temp^2+prec (county FE)')
b1=res['yield_quad']['gs_temp']['coef']; b2=res['yield_quad']['temp2']['coef']
thr=-b1/(2*b2) if b2!=0 else np.nan
res['yield_temp_optimum_C']=float(thr)
print('单产-温度最优阈值 ~ %.2f C'%thr)

# 3. 分纬度带
codes=p.groupby('CountyCode')['lat'].first().sort_values()
q=pd.qcut(codes,3,labels=['South','Central','North'])
band=dict(zip(codes.index,q))
p['band']=p['CountyCode'].map(band)
def band_stat(g):
    a85=g[g.year==g.year.min()]['paddy_ha'].sum(); a20=g[g.year==2020]['paddy_ha'].sum()
    t=g.groupby('year')['gs_temp'].mean()
    warm=np.polyfit(t.index,t.values,1)[0]*10
    return pd.Series({'paddy85_ha':a85,'paddy20_ha':a20,'expansion_x':a20/a85 if a85>0 else np.nan,'warming_C_dec':warm})
bs=p.groupby('band').apply(band_stat)
print('\n=== 分纬度带 ===\n',bs.to_string())
res['latitude_bands']=bs.reset_index().to_dict(orient='records')

# 每县扩张比与增温(供制图)
def cty(g):
    a85=g[g.year<=1987]['paddy_ha'].mean(); a20=g[g.year>=2018]['paddy_ha'].mean()
    t=g.dropna(subset=['gs_temp']); warm=np.polyfit(t.year,t.gs_temp,1)[0]*10 if len(t)>5 else np.nan
    y=g.dropna(subset=['yield_kg_ha'])
    y0=y[y.year<=1990]['yield_kg_ha'].mean(); y1=y[y.year>=2017]['yield_kg_ha'].mean()
    return pd.Series({'paddy_early':a85,'paddy_late':a20,'expansion_x':a20/a85 if a85>0 else np.nan,
                      'warming_C_dec':warm,'yield_early':y0,'yield_late':y1,
                      'lat':g['lat'].iloc[0],'lon':g['lon'].iloc[0]})
cc=p.groupby('CountyCode').apply(cty).reset_index()
cc=cc.merge(st[['CountyCode','Name']],on='CountyCode',how='left')
cc.to_csv(MID+'/county_change_summary.csv',index=False,encoding='utf-8-sig')

with open(MID+'/regression_results.json','w',encoding='utf-8') as f:
    json.dump(res,f,ensure_ascii=False,indent=2,default=float)
print('\nsaved regression_results.json & county_change_summary.csv')
