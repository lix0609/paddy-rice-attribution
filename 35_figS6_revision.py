# -*- coding: utf-8 -*-
"""SI 图 (修订支撑): (a)响应函数四形式敏感性 (b)逐年Moran's I (c)扩张潜力·情景演示(原Fig8c降级)。"""
import json,numpy as np,pandas as pd,warnings; warnings.filterwarnings('ignore')
import matplotlib as mpl; mpl.use('Agg'); import matplotlib.pyplot as plt
mpl.rcParams.update({'font.family':'serif','font.serif':['Times New Roman','Liberation Serif','Nimbus Roman','DejaVu Serif'],
 'font.weight':'bold','axes.labelweight':'bold','axes.titleweight':'bold','svg.fonttype':'none','pdf.fonttype':42,
 'font.size':7,'axes.titlesize':7.6,'axes.labelsize':7,'xtick.labelsize':6.2,'ytick.labelsize':6.2,'legend.fontsize':6,'axes.linewidth':0.8,'legend.frameon':False})
OI={'blue':'#0072B2','green':'#009E73','orange':'#E69F00','verm':'#D55E00'}; MM=1/25.4; DC=183*MM
PROJ='/sessions/vibrant-bold-franklin/mnt/水稻/增温驱动吉林水田扩张与增产_1985-2020'; MID=PROJ+'/02_中间数据'; OUTd=PROJ+'/09_投稿前修订_R1/02_outputs'; PUB=PROJ+'/03_figures/pub'
def plab(ax,L): ax.annotate(L,xy=(0,1),xycoords='axes fraction',xytext=(-24,6),textcoords='offset points',fontsize=9,fontweight='bold',va='bottom')
i3=json.load(open(OUTd+'/issue3_response_forms.json')); tr=json.load(open(MID+'/projection_trajectory.json'))
fig=plt.figure(figsize=(DC,DC*0.34),constrained_layout=True)
gsp=fig.add_gridspec(1,3,width_ratios=[1.35,1,1.28])
ax=[fig.add_subplot(gsp[0,0]),fig.add_subplot(gsp[0,1]),fig.add_subplot(gsp[0,2])]

# (a) 四形式敏感性 (mid/end 585)
forms=['linear','quadratic','piecewise','edd']; lab=['Linear','Quadratic','Piecewise','EDD']
mid=[i3['structural_band']['mid_585']['by_form'][f] for f in forms]
end=[i3['structural_band']['end_585']['by_form'][f] for f in forms]
x=np.arange(4); w=0.38
ax[0].bar(x-w/2,mid,w,color=OI['blue'],edgecolor='white',label='Mid-century')
ax[0].bar(x+w/2,end,w,color=OI['verm'],edgecolor='white',label='End-century')
ax[0].axhline(0,color='k',lw=0.6); ax[0].set_xticks(x); ax[0].set_xticklabels(lab,rotation=0,ha='center')
ax[0].set_ylabel('Climate-driven Δyield (%)'); ax[0].set_title('Response-form sensitivity (SSP5-8.5)'); ax[0].legend(loc='lower left')
plab(ax[0],'a')

# (b) 逐年 Moran's I (重算)
pan=pd.read_csv(MID+'/county_year_panel.csv'); pan['CountyCode']=pan['CountyCode'].astype(str)
F=pd.read_csv(MID+'/climate_features_county_year.csv'); F['CountyCode']=F['CountyCode'].astype(str)
d=pan.merge(F,on=['CountyCode','year'],how='left').dropna(subset=['yield_kg_ha','gs_tmean']); d=d[d.gdd>0].copy(); d['ln']=np.log(d.yield_kg_ha)
d=d.groupby('CountyCode',group_keys=False).apply(lambda g: g.assign(r=g.ln-np.polyval(np.polyfit(g.year,g.ln,1),g.year)) if g.year.nunique()>=8 else g.assign(r=np.nan)).dropna(subset=['r'])
def moran(sub):
    if len(sub)<8: return np.nan
    xy=sub[['lon','lat']].values; z=(sub.r-sub.r.mean()).values
    D=np.sqrt(((xy[:,None,:]-xy[None,:,:])**2).sum(-1)); np.fill_diagonal(D,np.inf)
    W=1/D; W[D>np.percentile(D[np.isfinite(D)],20)]=0; W=W/W.sum(1,keepdims=True); W[np.isnan(W)]=0
    return float((len(z)/W.sum())*((W*np.outer(z,z)).sum()/(z**2).sum()))
mi=sorted([(y,moran(g)) for y,g in d.groupby('year') if np.isfinite(moran(g))])
yy=[a for a,_ in mi]; vv=[b for _,b in mi]
ax[1].bar(yy,vv,color=OI['green'],width=0.8,edgecolor='white'); ax[1].axhline(0,color='k',lw=0.6)
ax[1].axhline(np.median(vv),color=OI['orange'],lw=1,ls='--',label='median %.2f'%np.median(vv))
ax[1].set_ylabel("Moran's I (yield residual)"); ax[1].set_xlabel('Year'); ax[1].set_title('Annual spatial autocorrelation'); ax[1].legend(loc='upper right')
plab(ax[1],'b')

# (c) 扩张潜力·情景演示 (原Fig8c降级; 单条,不与penalty并置,不画net)
def tget(ssp,win,k): return [t[k] for t in tr['trajectory'] if t['ssp_code']==ssp and t['window']==win][0]
xs=np.arange(2)
g245=[tget('245','2041','expansion_gain_Mt'),tget('245','2071','expansion_gain_Mt')]
g585=[tget('585','2041','expansion_gain_Mt'),tget('585','2071','expansion_gain_Mt')]
ax[2].bar(xs-0.2,g245,0.4,color=OI['blue'],edgecolor='white',label='SSP2-4.5')
ax[2].bar(xs+0.2,g585,0.4,color=OI['verm'],edgecolor='white',label='SSP5-8.5')
ax[2].set_xticks(xs); ax[2].set_xticklabels(['Mid-century','End-century']); ax[2].set_xlim(-0.65,1.65); ax[2].set_ylim(0,8.4)
ax[2].set_ylabel('Illustrative expansion potential (Mt yr$^{-1}$)')
ax[2].set_title('Illustrative scenario only'); ax[2].legend(loc='upper left')
ax[2].text(0.05,0.66,'illustrative: assumes historical\narea–T link persists; not a\nforecast, not netted vs yield',transform=ax[2].transAxes,fontsize=5,ha='left',va='top',color='0.45',style='italic')
plab(ax[2],'c')
for a in ax:
    for s in a.spines.values(): s.set_visible(True)
fig.savefig(PUB+'/FigS6_revision.pdf',bbox_inches='tight'); fig.savefig(PUB+'/FigS6_revision.png',dpi=600,bbox_inches='tight'); fig.savefig(PUB+'/FigS6_revision.svg',bbox_inches='tight'); plt.close()
print('FigS6 saved | Moran median=%.2f range[%.2f,%.2f]'%(np.median(vv),min(vv),max(vv)))
