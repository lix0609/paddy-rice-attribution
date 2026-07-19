import pandas as pd,numpy as np,json,warnings; warnings.filterwarnings('ignore')
from linearmodels.panel import PanelOLS
MID='02_中间数据'; PH='06_phenology'
ph=pd.read_csv(PH+'/phenology_station_year.csv')
cf=pd.read_csv(MID+'/climate_features_county_year.csv'); cf['CountyCode']=cf['CountyCode'].astype(str)
st=pd.read_csv(MID+'/county_static.csv'); st['CountyCode']=st['CountyCode'].astype(str)
# 站点匹配最近吉林县(haversine)
def hav(la1,lo1,la2,lo2):
    R=6371;p=np.pi/180
    a=np.sin((la2-la1)*p/2)**2+np.cos(la1*p)*np.cos(la2*p)*np.sin((lo2-lo1)*p/2)**2
    return 2*R*np.arcsin(np.sqrt(a))
rows=[]
for _,r in ph.iterrows():
    dmin=1e9;cc=None
    for _,c in st.iterrows():
        d=hav(r['lat'],r['lon'],c['lat'],c['lon'])
        if d<dmin: dmin=d;cc=c['CountyCode']
    rows.append((cc,dmin))
ph['CountyCode']=[x[0] for x in rows]; ph['dist_km']=[x[1] for x in rows]
ph=ph[ph.dist_km<50]  # 仅保留真正在吉林境内/邻近的站
ph=ph.merge(cf[['CountyCode','year','gs_tmean']],on=['CountyCode','year'],how='left').dropna(subset=['gs_tmean'])
print('匹配后可用: 记录%d 站点%d 县%d 年%d-%d'%(len(ph),ph.station.nunique(),ph.CountyCode.nunique(),ph.year.min(),ph.year.max()))

def panel(dep):
    d=ph.dropna(subset=[dep,'gs_tmean']).copy()
    if d.station.nunique()<3 or len(d)<30: return None
    d=d.set_index(['station','year'])
    try:
        r=PanelOLS(d[dep],d[['gs_tmean']],entity_effects=True,check_rank=False).fit(cov_type='clustered',cluster_entity=True)
        return dict(coef=float(r.params['gs_tmean']),p=float(r.pvalues['gs_tmean']),n=int(r.nobs))
    except Exception as e: return {'err':str(e)}
res={}
for dep,lab in [('heading','抽穗DOY'),('maturity','成熟DOY'),('gp_sow_mat','生育期(播种-成熟)'),('gp_trans_mat','生育期(移栽-成熟)'),('transplant','移栽DOY'),('sowing','播种DOY')]:
    if dep in ph.columns:
        r=panel(dep); res[dep]=r
        if r and 'coef' in r: print('%-18s ~ 生长季温度: %+.2f 天/°C  p=%.3f  n=%d'%(lab,r['coef'],r['p'],r['n']))
json.dump(res,open(PH+'/phenology_temp_sensitivity.json','w'),ensure_ascii=False,indent=2,default=float)
ph.to_csv(PH+'/phenology_matched_climate.csv',index=False,encoding='utf-8-sig')
print('saved.')
