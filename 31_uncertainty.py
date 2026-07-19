# -*- coding: utf-8 -*-
"""B2: 9个GCM的跨模式不确定性 (ΔY/适宜性/产量), 中叶+末世纪 × SSP245/585。"""
import numpy as np,pandas as pd,json,warnings; warnings.filterwarnings('ignore')
import statsmodels.api as sm
PROJ='/sessions/vibrant-bold-franklin/mnt/水稻/增温驱动吉林水田扩张与增产_1985-2020'; MID=PROJ+'/02_中间数据'
DIM={5:31,6:30,7:31,8:31,9:30}; CLIM=['gs_tmean','gdd','at10','t_warm','t_may','prec_gs','prec_78','prec_conc']; REQ=2840
def feats(tas,pr):
    gs=[5,6,7,8,9]
    return dict(gs_tmean=np.mean([tas[m] for m in gs]),t_warm=np.mean([tas[7],tas[8]]),t_may=tas[5],
        gdd=sum(max(tas[m]-10,0)*DIM[m] for m in gs),at10=sum(tas[m]*DIM[m] for m in gs if tas[m]>=10),
        prec_gs=sum(pr[m] for m in gs),prec_78=pr[7]+pr[8],prec_conc=(pr[7]+pr[8])/sum(pr[m] for m in gs))
# 观测二次响应
pan=pd.read_csv(MID+'/county_year_panel.csv'); pan['CountyCode']=pan['CountyCode'].astype(str)
F=pd.read_csv(MID+'/climate_features_county_year.csv'); F['CountyCode']=F['CountyCode'].astype(str)
df=pan.merge(F,on=['CountyCode','year'],how='left').dropna(subset=['yield_kg_ha']+CLIM); df=df[df.gdd>0].copy(); df['ln']=np.log(df.yield_kg_ha)
df=df.groupby('CountyCode',group_keys=False).apply(lambda g: g.assign(r=g.ln-np.polyval(np.polyfit(g.year,g.ln,1),g.year)) if g.year.nunique()>=8 else g.assign(r=np.nan)).dropna(subset=['r'])
modq=sm.OLS(df.r,sm.add_constant(pd.DataFrame({'gs':df.gs_tmean,'tw':df.t_warm,'tw2':df.t_warm**2,'pr':df.prec_gs}))).fit()
def predq(gs,tw,pr): return modq.predict(sm.add_constant(pd.DataFrame({'gs':gs,'tw':tw,'tw2':tw**2,'pr':pr}),has_constant='add'))
obsB=F[(F.year>=1991)&(F.year<=2020)&(F.gdd>0)].groupby('CountyCode')[CLIM].mean()
recent=pan[(pan.year>=2016)&(pan.year<=2020)].groupby('CountyCode').agg(prod_kg=('prod_kg','mean'))
# 基准(CMIP ESM base)
allw=pd.read_csv(MID+'/cmip6_county_monthly_allwin.csv'); allw['CountyCode']=allw['CountyCode'].astype(str)
dtb=allw[allw.dataset=='tas_base'].set_index('CountyCode'); dpb=allw[allw.dataset=='pr_base'].set_index('CountyCode')
codesB=[c for c in dtb.index if c in dpb.index]
BASEc=pd.DataFrame({c:feats({m:dtb.loc[c,'m%02d'%m] for m in range(1,13)},{m:dpb.loc[c,'m%02d'%m] for m in range(1,13)}) for c in codesB}).T[CLIM]
# GCM 月度
gcm=pd.read_csv(MID+'/cmip6_gcm_monthly.csv'); gcm['CountyCode']=gcm['CountyCode'].astype(str)
MODELS=sorted(set(d.split('_')[1] for d in gcm.dataset.unique()))
print('模式(%d):'%len(MODELS),MODELS)
def model_feats(model,win,ssp):
    dt=gcm[gcm.dataset=='tas_%s_%s_%s'%(model,win,ssp)].set_index('CountyCode')
    dp=gcm[gcm.dataset=='pr_%s_%s_%s'%(model,win,ssp)].set_index('CountyCode')
    codes=[c for c in dt.index if c in dp.index]
    return pd.DataFrame({c:feats({m:dt.loc[c,'m%02d'%m] for m in range(1,13)},{m:dp.loc[c,'m%02d'%m] for m in range(1,13)}) for c in codes}).T[CLIM]
res={}; percounty={}
for win in ['2041','2071']:
    for ssp in ['245','585']:
        dys=[]; suits=[]; losses=[]; dTs=[]; percty=[]
        for mo in MODELS:
            Fw=model_feats(mo,win,ssp); idx=[c for c in obsB.index if c in Fw.index and c in BASEc.index]
            delta=Fw.loc[idx]-BASEc.loc[idx]; base=obsB.loc[idx]; proj=base+delta
            dY=100*(predq(proj.gs_tmean,proj.t_warm,proj.prec_gs).values-predq(base.gs_tmean,base.t_warm,base.prec_gs).values)
            dys.append(float(np.nanmean(dY))); suits.append(100*float((proj.at10>=REQ).mean()))
            prod_t=(recent['prod_kg'].reindex(idx)/1000).values
            losses.append(float(np.nansum(prod_t*dY/100))/1e3)  # kt
            dTs.append(float((proj.gs_tmean-base.gs_tmean).mean())); percty.append(pd.Series(dY,index=idx))
        dys=np.array(dys); key='%s_%s'%(win,ssp)
        # 逐县: 模式一致减产比例
        PC=pd.concat(percty,axis=1); agree_cty=float((PC<0).all(axis=1).mean())
        res[key]={'models':MODELS,'dY_models':[float(x) for x in dys],
          'dY_mean':float(dys.mean()),'dY_sd':float(dys.std(ddof=1)),'dY_min':float(dys.min()),'dY_max':float(dys.max()),
          'dT_mean':float(np.mean(dTs)),'suit_mean':float(np.mean(suits)),'suit_min':float(np.min(suits)),'suit_max':float(np.max(suits)),
          'loss_kt_mean':float(np.mean(losses)),'loss_kt_sd':float(np.std(losses,ddof=1)),'loss_kt_min':float(np.min(losses)),'loss_kt_max':float(np.max(losses)),
          'sign_agree_prov':float((dys<0).mean()),'sign_agree_county_all':agree_cty}
        percounty[key]=PC.mean(axis=1)
json.dump(res,open(MID+'/projection_uncertainty.json','w'),ensure_ascii=False,indent=2,default=float)
print('\n=== 跨模式ΔY (均值 [极差], 模式SD, 省级符号一致) ===')
for win,yr in [('2041','中叶'),('2071','末世纪')]:
    for ssp in ['245','585']:
        r=res['%s_%s'%(win,ssp)]
        print(' %s SSP%s: ΔY=%.2f%% [%.2f,%.2f] SD=%.2f | 减产吨%.0f[%.0f,%.0f]kt | %d/9模式减产 逐县全一致%.0f%%'%(
          yr,ssp,r['dY_mean'],r['dY_min'],r['dY_max'],r['dY_sd'],r['loss_kt_mean'],r['loss_kt_min'],r['loss_kt_max'],
          round(r['sign_agree_prov']*9),100*r['sign_agree_county_all']))
print('\nsaved projection_uncertainty.json')
