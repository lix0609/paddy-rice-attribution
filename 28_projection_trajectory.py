# -*- coding: utf-8 -*-
"""方案②(Tier1): 三窗(2030/2055/2085)x SSP245/585 轨迹 + 末世纪定量 (ESM集合均值)。"""
import numpy as np,pandas as pd,json,warnings; warnings.filterwarnings('ignore')
import statsmodels.api as sm
PROJ='/sessions/vibrant-bold-franklin/mnt/水稻/增温驱动吉林水田扩张与增产_1985-2020'; MID=PROJ+'/02_中间数据'
DIM={5:31,6:30,7:31,8:31,9:30}; CLIM=['gs_tmean','gdd','at10','t_warm','t_may','prec_gs','prec_78','prec_conc']
REQ=2840; BETA=0.23379845441467773
def feats(tas,pr):
    gs=[5,6,7,8,9]
    return dict(gs_tmean=np.mean([tas[m] for m in gs]),t_warm=np.mean([tas[7],tas[8]]),t_may=tas[5],
        gdd=sum(max(tas[m]-10,0)*DIM[m] for m in gs),at10=sum(tas[m]*DIM[m] for m in gs if tas[m]>=10),
        prec_gs=sum(pr[m] for m in gs),prec_78=pr[7]+pr[8],prec_conc=(pr[7]+pr[8])/sum(pr[m] for m in gs))
# 观测拟合二次响应
pan=pd.read_csv(MID+'/county_year_panel.csv'); pan['CountyCode']=pan['CountyCode'].astype(str)
F=pd.read_csv(MID+'/climate_features_county_year.csv'); F['CountyCode']=F['CountyCode'].astype(str)
df=pan.merge(F,on=['CountyCode','year'],how='left').dropna(subset=['yield_kg_ha']+CLIM); df=df[df.gdd>0].copy()
df['ln']=np.log(df.yield_kg_ha)
df=df.groupby('CountyCode',group_keys=False).apply(lambda g: g.assign(r=g.ln-np.polyval(np.polyfit(g.year,g.ln,1),g.year)) if g.year.nunique()>=8 else g.assign(r=np.nan)).dropna(subset=['r'])
modq=sm.OLS(df.r,sm.add_constant(pd.DataFrame({'gs':df.gs_tmean,'tw':df.t_warm,'tw2':df.t_warm**2,'pr':df.prec_gs}))).fit()
def predq(gs,tw,pr): return modq.predict(sm.add_constant(pd.DataFrame({'gs':gs,'tw':tw,'tw2':tw**2,'pr':pr}),has_constant='add'))
obsB=F[(F.year>=1991)&(F.year<=2020)&(F.gdd>0)].groupby('CountyCode')[CLIM].mean()
# CMIP6 多窗
cm=pd.read_csv(MID+'/cmip6_county_monthly_allwin.csv'); cm['CountyCode']=cm['CountyCode'].astype(str)
def getf(ds):
    d=cm[cm.dataset==ds].set_index('CountyCode')
    return pd.DataFrame({c:feats({m:d.loc[c,'m%02d'%m] for m in range(1,13)},{m:d.loc[c,'m%02d'%m] for m in range(1,13)}) for c in d.index}).T[CLIM]
# 注意: tas 与 pr 在同一 dataset? 不, 分开. 需分别取 tas_* 与 pr_* 组月温/月雨
def get_tp(win,ssp):
    dt=cm[cm.dataset=='tas_%s_%s'%(ssp,win)].set_index('CountyCode'); dp=cm[cm.dataset=='pr_%s_%s'%(ssp,win)].set_index('CountyCode')
    codes=[c for c in dt.index if c in dp.index]; out={}
    for c in codes:
        tas={m:dt.loc[c,'m%02d'%m] for m in range(1,13)}; pr={m:dp.loc[c,'m%02d'%m] for m in range(1,13)}
        out[c]=feats(tas,pr)
    return pd.DataFrame(out).T[CLIM]
dtb=cm[cm.dataset=='tas_base'].set_index('CountyCode'); dpb=cm[cm.dataset=='pr_base'].set_index('CountyCode')
codesB=[c for c in dtb.index if c in dpb.index]
BASEc=pd.DataFrame({c:feats({m:dtb.loc[c,'m%02d'%m] for m in range(1,13)},{m:dpb.loc[c,'m%02d'%m] for m in range(1,13)}) for c in codesB}).T[CLIM]
recent=pan[(pan.year>=2016)&(pan.year<=2020)].groupby('CountyCode').agg(prod_kg=('prod_kg','mean'),paddy_ha=('paddy_ha','mean'),yield_kg_ha=('yield_kg_ha','mean'))
st=pan.groupby('CountyCode')['lat'].mean(); Name=pan.groupby('CountyCode')['Name'].first()
WINS={'2021':2030,'2041':2055,'2071':2085}
traj=[]; endc={}
for win,yr in WINS.items():
    for ssp in ['245','585']:
        Fw=get_tp(win,ssp); idx=[c for c in obsB.index if c in Fw.index and c in BASEc.index]
        delta=Fw.loc[idx]-BASEc.loc[idx]; base=obsB.loc[idx]; proj=base+delta
        dY=100*(predq(proj.gs_tmean,proj.t_warm,proj.prec_gs).values-predq(base.gs_tmean,base.t_warm,base.prec_gs).values)
        suit=100*(proj.at10>=REQ).mean()
        prod_t=(recent['prod_kg'].reindex(idx)/1000).values
        loss_t=float(np.nansum(prod_t*dY/100)); prov_t=float(np.nansum(prod_t))
        dTgs=float((proj.gs_tmean-base.gs_tmean).mean())
        gain_t=float(recent['paddy_ha'].reindex(idx).sum()*BETA*dTgs*recent['yield_kg_ha'].reindex(idx).mean()/1000)
        traj.append(dict(window=win,mid_year=yr,ssp={'245':'SSP2-4.5','585':'SSP5-8.5'}[ssp],ssp_code=ssp,
            dT_gs=dTgs,dY_mean=float(np.nanmean(dY)),suitability=suit,
            loss_kt=loss_t/1e3,loss_pct=100*loss_t/prov_t,n_loss=int((dY<0).sum()),n=len(idx),
            expansion_gain_Mt=gain_t/1e6,net_Mt=(gain_t+loss_t)/1e6))
        if win=='2071':
            r=pd.DataFrame({'CountyCode':idx,'Name':Name.reindex(idx).values,'lat':st.reindex(idx).values,'dY':dY,
                            't_warm':proj.t_warm.values,'prod_t':prod_t}); q1,q2=r.lat.quantile([1/3,2/3])
            r['band']=np.where(r.lat<=q1,'South',np.where(r.lat<=q2,'Central','North'))
            bands={b:{'dY_mean':float(r[r.band==b].dY.mean()),'n_loss':int((r[r.band==b].dY<0).sum()),'n':int((r.band==b).sum())} for b in ['North','Central','South']}
            w5=r.nsmallest(5,'dY')[['Name','band','dY','t_warm']].round(2)
            endc[ssp]={'dY_mean':float(np.nanmean(dY)),'n_loss':int((dY<0).sum()),'suitability':suit,
                       'loss_kt':loss_t/1e3,'loss_pct':100*loss_t/prov_t,'bands':bands,
                       'most_exposed':w5.to_dict('records'),'worst':float(dY.min())}
            r.to_csv(MID+'/projection_county_2071_%s.csv'%ssp,index=False,encoding='utf-8-sig')
T=pd.DataFrame(traj)
print('=== 轨迹 (省均ΔY% / 适宜性% / 减产吨kt) ===')
print(T[['window','mid_year','ssp','dT_gs','dY_mean','suitability','loss_kt','n_loss']].round(2).to_string(index=False))
print('\n=== 末世纪(2071-2100) 分带 ===')
for ssp in ['245','585']:
    print(' SSP%s:'%ssp, {b:(round(endc[ssp]["bands"][b]["dY_mean"],2),endc[ssp]["bands"][b]["n_loss"]) for b in endc[ssp]['bands']})
json.dump({'trajectory':traj,'endcentury':endc,'windows':WINS},open(MID+'/projection_trajectory.json','w'),ensure_ascii=False,indent=2,default=float)
print('\nsaved projection_trajectory.json + projection_county_2071_*.csv')
