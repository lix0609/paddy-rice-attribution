# -*- coding: utf-8 -*-
import os, glob, re, json, warnings
import numpy as np, pandas as pd
warnings.filterwarnings('ignore')
ROOT='/sessions/vibrant-bold-franklin/mnt/水稻'
MID=ROOT+'/增温驱动吉林水田扩张与增产_1985-2020/02_中间数据'
os.makedirs(MID,exist_ok=True)

# ---- 1. 水田面积(遥感) ----
pa=pd.read_csv(MID+'/paddy_area_by_county_1985_2020.csv')
st=pd.read_csv(MID+'/county_static.csv')
st['CountyCode']=st['CountyCode'].astype(str); pa['CountyCode']=pa['CountyCode'].astype(str)
pa=pa.merge(st[['CountyCode','lon','lat','elev_mean']],on='CountyCode',how='left')

prov_map=pa.groupby('year')['paddy_ha'].sum().reset_index().rename(columns={'paddy_ha':'paddy_ha_RS'})

# 面积加权质心/纬度/高程迁移
def wmean(g,col,w='paddy_ha'):
    return np.average(g[col],weights=g[w]) if g[w].sum()>0 else np.nan
mig=pa.groupby('year').apply(lambda g: pd.Series({
    'cent_lon':wmean(g,'lon'),'cent_lat':wmean(g,'lat'),
    'mean_elev':wmean(g,'elev_mean')})).reset_index()
mig.to_csv(MID+'/centroid_migration.csv',index=False,encoding='utf-8-sig')

# ---- 2. 产量面板(年鉴) ----
rows=[]
for f in glob.glob(ROOT+'/产量数据/整理后的产量数据/Jilin_Rice_Panel_Data_*.csv'):
    m=re.search(r'(\d{4})',os.path.basename(f))
    if not m: continue
    y=int(m.group(1))
    if not (1985<=y<=2020): continue
    try: d=pd.read_csv(f,encoding='utf-8-sig')
    except: d=pd.read_csv(f,encoding='gbk')
    d.columns=[c.strip() for c in d.columns]
    ac=[c for c in d.columns if '面积' in c][0]; pc=[c for c in d.columns if '产量' in c][0]
    A=pd.to_numeric(d[ac],errors='coerce').sum(); P=pd.to_numeric(d[pc],errors='coerce').sum()
    rows.append((y,A,P))
yb=pd.DataFrame(rows,columns=['year','area_ha','prod_kg']).sort_values('year')
# 清洗1995面积异常(单位错), 用产量/合理单产回推或置NaN
yb.loc[yb['area_ha']>3_000_000,'area_ha']=np.nan
yb['yield_kg_ha']=yb['prod_kg']/yb['area_ha']
# 清洗不合理单产(录入错误): 水稻单产应在3000-13000 kg/ha
bad=(yb['yield_kg_ha']<3000)|(yb['yield_kg_ha']>13000)
yb.loc[bad,['area_ha','prod_kg','yield_kg_ha']]=np.nan
yb.to_csv(MID+'/yield_panel_province.csv',index=False,encoding='utf-8-sig')

# ---- 3. LMDI 分解: 产量变化 = 面积效应 + 单产效应 (加法LMDI) ----
def lmdi(P0,A0,Y0,P1,A1,Y1):
    L=lambda a,b:(b-a)/(np.log(b)-np.log(a)) if (a>0 and b>0 and a!=b) else (a if a==b else np.nan)
    Lw=L(P0,P1)
    dA=Lw*np.log(A1/A0); dY=Lw*np.log(Y1/Y0)
    return dA,dY
ybc=yb.dropna(subset=['area_ha','prod_kg','yield_kg_ha'])
y0=ybc[ybc.year==ybc.year.min()].iloc[0]; y1=ybc[ybc.year==ybc.year.max()].iloc[0]
dA,dY=lmdi(y0.prod_kg,y0.area_ha,y0.yield_kg_ha,y1.prod_kg,y1.area_ha,y1.yield_kg_ha)
tot=y1.prod_kg-y0.prod_kg
lmdi_res={'period':f'{int(y0.year)}-{int(y1.year)}','total_dProd_kg':tot,
          'area_effect_kg':dA,'yield_effect_kg':dY,
          'area_share_%':100*dA/tot,'yield_share_%':100*dY/tot}

# ---- 4. 气候趋势 ----
gt=pd.read_excel(ROOT+'/climate_output/growing_temp_1985_2020.xlsx')
gp=pd.read_excel(ROOT+'/climate_output/growing_prec_1985_2020.xlsx')
yrs=[c for c in gt.columns if str(c).isdigit()]
temp_prov=gt[yrs].mean(axis=0); prec_prov=gp[yrs].mean(axis=0)
clim=pd.DataFrame({'year':[int(y) for y in yrs],'gs_temp':temp_prov.values,'gs_prec':prec_prov.values})
clim.to_csv(MID+'/climate_province.csv',index=False,encoding='utf-8-sig')
def sen_trend(x,y):
    x=np.asarray(x,float);y=np.asarray(y,float);m=~np.isnan(y)
    x,y=x[m],y[m]
    from itertools import combinations
    s=[(y[j]-y[i])/(x[j]-x[i]) for i,j in combinations(range(len(x)),2)]
    return np.median(s)
temp_slope=sen_trend(clim.year,clim.gs_temp); prec_slope=sen_trend(clim.year,clim.gs_prec)

# ---- 汇总 ----
def val(df,col,yr): 
    r=df[df.year==yr]; return float(r[col].iloc[0]) if len(r) else np.nan
head={
 'RS_paddy_ha_1985':val(prov_map,'paddy_ha_RS',1985),
 'RS_paddy_ha_2020':val(prov_map,'paddy_ha_RS',2020),
 'RS_expansion_x':val(prov_map,'paddy_ha_RS',2020)/val(prov_map,'paddy_ha_RS',1985),
 'YB_area_1985':val(yb,'area_ha',1985),'YB_area_2020':val(yb,'area_ha',2020),
 'YB_prod_1985':val(yb,'prod_kg',1985),'YB_prod_2020':val(yb,'prod_kg',2020),
 'YB_yield_1985':val(yb,'yield_kg_ha',1985),'YB_yield_2020':val(yb,'yield_kg_ha',2020),
 'cent_lat_1985':val(mig,'cent_lat',1985),'cent_lat_2020':val(mig,'cent_lat',2020),
 'mean_elev_1985':val(mig,'mean_elev',1985),'mean_elev_2020':val(mig,'mean_elev',2020),
 'gs_temp_slope_C_per_yr':temp_slope,'gs_prec_slope_mm_per_yr':prec_slope,
 'LMDI':lmdi_res}
with open(MID+'/headline_numbers.json','w',encoding='utf-8') as f:
    json.dump(head,f,ensure_ascii=False,indent=2,default=float)

print('=== 省级水田面积(遥感) 关键年 ===')
print(prov_map[prov_map.year.isin([1985,1990,2000,2010,2020])].to_string(index=False))
print('\n=== 质心/纬度/高程迁移 关键年 ===')
print(mig[mig.year.isin([1985,2000,2020])].to_string(index=False))
print('\n=== LMDI ===');print(json.dumps(lmdi_res,ensure_ascii=False,indent=2,default=float))
print('\n=== 气候趋势 ===');print('温度Sen斜率 %.4f C/yr, 降水 %.3f mm/yr'%(temp_slope,prec_slope))
print('温度: 1985 %.2f -> 2020 %.2f'%(val(clim,'gs_temp',1985),val(clim,'gs_temp',2020)))
print('\n=== HEADLINE ===');print(json.dumps(head,ensure_ascii=False,indent=2,default=float))
