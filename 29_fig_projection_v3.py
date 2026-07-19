# -*- coding: utf-8 -*-
"""Fig 7 v3: 多窗轨迹 + 末世纪 (mid vs late century)。"""
import json,numpy as np,pandas as pd,geopandas as gpd,warnings; warnings.filterwarnings('ignore')
import matplotlib as mpl; mpl.use('Agg'); import matplotlib.pyplot as plt
mpl.rcParams.update({'font.family':'serif','font.serif':['Times New Roman','Liberation Serif','Nimbus Roman','DejaVu Serif'],
 'font.weight':'bold','axes.labelweight':'bold','axes.titleweight':'bold','svg.fonttype':'none','pdf.fonttype':42,
 'font.size':7,'axes.titlesize':7.8,'axes.labelsize':7,'xtick.labelsize':6.3,'ytick.labelsize':6.3,'legend.fontsize':6,
 'axes.linewidth':0.8,'legend.frameon':False})
OI={'blue':'#0072B2','green':'#009E73','orange':'#E69F00','verm':'#D55E00'}; MM=1/25.4; DC=183*MM
ROOT='/sessions/vibrant-bold-franklin/mnt/水稻'; PROJ=ROOT+'/增温驱动吉林水田扩张与增产_1985-2020'; MID=PROJ+'/02_中间数据'; PUB=PROJ+'/03_figures/pub'
def plab(ax,L): ax.annotate(L,xy=(0,1),xycoords='axes fraction',xytext=(-20,5),textcoords='offset points',fontsize=9,fontweight='bold',va='bottom')
J=json.load(open(MID+'/projection_trajectory.json')); traj=J['trajectory']
U=json.load(open(MID+'/projection_uncertainty.json'))
def tget(ssp,win,k): return [t[k] for t in traj if t['ssp_code']==ssp and t['window']==win][0]
g=gpd.read_file(ROOT+'/吉林省_区县行政边界/吉林省_区县_合并50.shp').to_crs(4326); g['CountyCode']=g['CountyCode'].astype(str)
mid=pd.read_csv(MID+'/projection_county_v2.csv'); mid['CountyCode']=mid['CountyCode'].astype(str)
end=pd.read_csv(MID+'/projection_county_2071_585.csv'); end['CountyCode']=end['CountyCode'].astype(str)
mmid=g.merge(mid[['CountyCode','dY585']],on='CountyCode',how='left')
mend=g.merge(end[['CountyCode','dY']],on='CountyCode',how='left')
fig,ax=plt.subplots(2,3,figsize=(DC,DC*0.70),constrained_layout=True)
vmax=float(max(np.nanmax(np.abs(mid.dY585)),np.nanmax(np.abs(end.dY))))
for a,mp,col,ttl,L in [(ax[0,0],mmid,'dY585','Mid-century 2041–2070','a'),(ax[0,1],mend,'dY','End-century 2071–2100','b')]:
    mp.plot(column=col,ax=a,cmap='RdBu',vmin=-vmax,vmax=vmax,edgecolor='0.5',lw=0.25,legend=True,legend_kwds={'label':'Climate-driven Δyield (%)','shrink':0.62})
    a.set_title('SSP5-8.5 yield change:\n'+ttl); a.axis('off'); plab(a,L)

# (c) trajectory ΔY + suitability
axc=ax[0,2]; yrs=[2020,2030,2055,2085]
for ssp,ccol in [('245',OI['blue']),('585',OI['verm'])]:
    # 9-模式 ±1SD 误差带 (仅中叶2055/末世纪2085有GCM)
    for w,x in [('2041',2055),('2071',2085)]:
        u=U['%s_%s'%(w,ssp)]; lo=u['dY_mean']-u['dY_sd']; hi=u['dY_mean']+u['dY_sd']
        axc.plot([x,x],[lo,hi],color=ccol,lw=1.0,alpha=0.55,zorder=1,solid_capstyle='round')
        for yy in (lo,hi): axc.plot([x-2,x+2],[yy,yy],color=ccol,lw=0.7,alpha=0.55)
    dy=[0]+[tget(ssp,w,'dY_mean') for w in ['2021','2041','2071']]
    axc.plot(yrs,dy,'-o',color=ccol,ms=4,lw=1.4,zorder=3,label='SSP2-4.5' if ssp=='245' else 'SSP5-8.5')
axc.axhline(0,color='k',lw=0.5); axc.set_ylabel('Climate-driven Δyield (%)'); axc.set_xlabel('Year (window mid-point)')
axc.set_ylim(-13.2,1.8); axc.set_xticks(yrs); axc.legend(loc='lower left')
ax2=axc.twinx()
for ssp,ccol in [('245',OI['blue']),('585',OI['verm'])]:
    su=[80]+[tget(ssp,w,'suitability') for w in ['2021','2041','2071']]
    ax2.plot(yrs,su,'--s',color=ccol,ms=3,lw=0.9,alpha=0.6)
ax2.set_ylabel('Thermally suitable counties (%)',color='0.35'); ax2.set_ylim(72,112); ax2.tick_params(colors='0.35')
axc.set_title('Divergence widens with time'); plab(axc,'c')

# (d) production penalty kt by window x scenario
axd=ax[1,0]; wl=['2021','2041','2071']; xw=np.arange(3); w=0.38
l245=[tget('245',x,'loss_kt') for x in wl]; l585=[tget('585',x,'loss_kt') for x in wl]
e245=[0]+[U['%s_245'%x]['loss_kt_sd'] for x in ['2041','2071']]; e585=[0]+[U['%s_585'%x]['loss_kt_sd'] for x in ['2041','2071']]
axd.bar(xw-w/2,l245,w,color=OI['blue'],edgecolor='white',label='SSP2-4.5',yerr=e245,error_kw=dict(lw=0.7,ecolor='0.3',capsize=2))
axd.bar(xw+w/2,l585,w,color=OI['verm'],edgecolor='white',label='SSP5-8.5',yerr=e585,error_kw=dict(lw=0.7,ecolor='0.3',capsize=2))
axd.axhline(0,color='k',lw=0.6); axd.set_xticks(xw); axd.set_xticklabels(['2030','2055','2085'])
axd.set_ylabel('Climate-driven production\nchange (kt yr$^{-1}$)'); axd.set_xlabel('Window mid-point'); axd.legend(loc='lower left')
for xi,v in zip(xw+w/2,l585): axd.text(xi,v-12,f'{v:.0f}',ha='center',va='top',fontsize=5.6,color=OI['verm'])
axd.set_title('Penalty deepens toward 2085'); plab(axd,'d')

# (e) net outlook mid vs end (SSP5-8.5)
axe=ax[1,1]; xs=np.arange(2)
gain=[tget('585','2041','expansion_gain_Mt'),tget('585','2071','expansion_gain_Mt')]
pen=[tget('585','2041','loss_kt')/1e3,tget('585','2071','loss_kt')/1e3]
net=[tget('585','2041','net_Mt'),tget('585','2071','net_Mt')]
axe.bar(xs-0.2,gain,0.4,color=OI['green'],edgecolor='white',label='Expansion (+)')
axe.bar(xs+0.2,pen,0.4,color=OI['verm'],edgecolor='white',label='Penalty (−)')
axe.plot(xs,net,'D',color='k',ms=5,zorder=4,label='Net')
axe.axhline(0,color='k',lw=0.6); axe.set_xticks(xs); axe.set_xticklabels(['Mid-century','End-century'])
axe.set_ylabel('Climate-attributable\nproduction (Mt yr$^{-1}$)'); axe.set_ylim(-1.2,7.4); axe.legend(loc='upper left',fontsize=5.4)
for xi,v in zip(xs-0.2,gain): axe.text(xi,v+0.15,f'+{v:.1f}',ha='center',va='bottom',fontsize=5.8,color=OI['green'])
for xi,v in zip(xs,net): axe.text(xi+0.26,v,f'net {v:+.1f}',ha='left',va='center',fontsize=5.6)
axe.set_title('SSP5-8.5: expansion vs penalty'); plab(axe,'e')

# (f) heat scatter end-century 585
axf=ax[1,2]; axf.axhline(0,color='0.6',lw=0.6); axf.axvline(21.9,color=OI['orange'],lw=1,ls=':')
axf.scatter(end.t_warm,end.dY,s=14,color=OI['verm'],edgecolor='white',lw=0.3)
ex=end.nsmallest(5,'dY'); axf.scatter(ex.t_warm,ex.dY,s=42,facecolors='none',edgecolors='k',lw=0.9,zorder=5)
axf.set_xlabel('End-century warmest-month T (°C)'); axf.set_ylabel('Climate-driven Δyield (%)')
axf.set_title('End-century heat penalty (SSP5-8.5)'); plab(axf,'f')
yl=axf.get_ylim()
axf.annotate('optimum 21.9 °C',xy=(21.9,yl[0]*0.45),xytext=(22.6,yl[0]*0.28),fontsize=6,color=OI['orange'],arrowprops=dict(arrowstyle='-',color=OI['orange'],lw=0.6))
axf.annotate('most-exposed\nwestern plains',xy=(ex.t_warm.iloc[0],ex.dY.iloc[0]),xytext=(26.5,-2.2),fontsize=5.8,ha='center',va='center',arrowprops=dict(arrowstyle='->',lw=0.6,color='0.3'))
for a in [axc,axd,axe,axf]:
    for s in a.spines.values(): s.set_visible(True)
fig.savefig(PUB+'/Fig11_projection.pdf',bbox_inches='tight'); fig.savefig(PUB+'/Fig11_projection.png',dpi=600,bbox_inches='tight'); fig.savefig(PUB+'/Fig11_projection.svg',bbox_inches='tight'); plt.close()
print('Fig7 v3 saved')
