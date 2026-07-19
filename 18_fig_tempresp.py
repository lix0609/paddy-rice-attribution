import pandas as pd,numpy as np,warnings; warnings.filterwarnings('ignore')
import matplotlib as mpl; mpl.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
mpl.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial','DejaVu Sans'],'svg.fonttype':'none','pdf.fonttype':42,
 'font.size':7,'axes.titlesize':7.8,'axes.labelsize':7,'xtick.labelsize':6.3,'ytick.labelsize':6.3,'axes.linewidth':0.7,
 'axes.spines.top':False,'axes.spines.right':False})
OI={'blue':'#0072B2','green':'#009E73','orange':'#E69F00','verm':'#D55E00'}; MM=1/25.4; DC=183*MM
ROOT='/sessions/vibrant-bold-franklin/mnt/水稻'; MID=ROOT+'/增温驱动吉林水田扩张与增产_1985-2020/02_中间数据'; PUB=ROOT+'/增温驱动吉林水田扩张与增产_1985-2020/03_figures/pub'
def plab(ax,L): ax.annotate(L,xy=(0,1),xycoords='axes fraction',xytext=(-32,6),textcoords='offset points',fontsize=9,fontweight='bold',va='bottom')
pan=pd.read_csv(MID+'/county_year_panel.csv'); pan['CountyCode']=pan['CountyCode'].astype(str)
F=pd.read_csv(MID+'/climate_features_county_year.csv'); F['CountyCode']=F['CountyCode'].astype(str)
CLIM=['gs_tmean','t_warm','gdd']
df=pan.merge(F,on=['CountyCode','year'],how='left').dropna(subset=['yield_kg_ha']+CLIM); df=df[df.gdd>0].copy(); df['ln_yield']=np.log(df['yield_kg_ha'])
df=df.groupby('CountyCode',group_keys=False).apply(lambda g: g.assign(r=(g['ln_yield']-np.polyval(np.polyfit(g.year,g.ln_yield,1),g.year)) if g.year.nunique()>=8 else np.nan)).dropna(subset=['r'])
df['ya']=100*df['r']
def binresp(col,step):
    lo,hi=np.floor(df[col].min()),np.ceil(df[col].max()); bins=np.arange(lo,hi+step,step)
    df['_b']=pd.cut(df[col],bins); g=df.groupby('_b')['ya'].agg(['mean','sem','count'])
    g=g[g['count']>=15]; g['ctr']=[iv.mid for iv in g.index]; return g
fig,ax=plt.subplots(1,2,figsize=(DC*0.74,DC*0.34),constrained_layout=True)
# a: GS mean temp
g=binresp('gs_tmean',0.5)
ax[0].axhline(0,color='0.6',lw=0.6)
ax[0].errorbar(g.ctr,g['mean'],yerr=g['sem'],fmt='o-',color=OI['blue'],ms=3.5,mec='white',mew=0.4,capsize=2,lw=1)
xf=np.linspace(g.ctr.min(),g.ctr.max(),100); b=np.polyfit(df['gs_tmean'],df['ya'],2); ax[0].plot(xf,np.polyval(b,xf),'--',color='0.4',lw=1)
best=g['mean'].idxmax(); ax[0].set_xlabel('Growing-season mean temperature (°C)'); ax[0].set_ylabel('Climate-driven yield anomaly (%)')
ax[0].set_title('Growing-season mean T')
ax[0].text(0.04,0.93,'Higher T → higher yield\n(cold-region benefit)',transform=ax[0].transAxes,fontsize=6,va='top',color=OI['blue']); plab(ax[0],'a')
# b: warmest-month temp with optimum
g=binresp('t_warm',0.5)
ax[1].axhline(0,color='0.6',lw=0.6)
ax[1].errorbar(g.ctr,g['mean'],yerr=g['sem'],fmt='o-',color=OI['verm'],ms=3.5,mec='white',mew=0.4,capsize=2,lw=1)
b=np.polyfit(df['t_warm'],df['ya'],2); opt=-b[1]/(2*b[0]); xf=np.linspace(g.ctr.min(),g.ctr.max(),100); ax[1].plot(xf,np.polyval(b,xf),'--',color='0.4',lw=1)
ax[1].axvline(opt,color=OI['orange'],lw=1,ls=':'); 
ax[1].set_xlabel('Warmest-month temperature (°C)'); ax[1].set_ylabel('Climate-driven yield anomaly (%)')
ax[1].set_title('Warmest-month T: optimum')
ax[1].annotate(f'optimum ≈ {opt:.1f}°C',xy=(opt,np.polyval(b,opt)),xytext=(opt-3.2,3.6),fontsize=6,color=OI['orange'],
    arrowprops=dict(arrowstyle='->',color=OI['orange'],lw=0.7))
ax[1].text(0.62,0.10,'yield falls\nbeyond ~24°C',transform=ax[1].transAxes,fontsize=6,color=OI['verm'],va='bottom'); plab(ax[1],'b')
fig.savefig(PUB+'/Fig8_temp_response.pdf',bbox_inches='tight'); fig.savefig(PUB+'/Fig8_temp_response.png',dpi=600,bbox_inches='tight'); fig.savefig(PUB+'/Fig8_temp_response.svg',bbox_inches='tight'); plt.close()
print('Fig8_temp_response done, warmest-month optimum=%.1f'%opt)

# ===== 数据一致性 SI 图 =====
pa=pd.read_csv(MID+'/paddy_area_by_county_1985_2020.csv'); pa['CountyCode']=pa['CountyCode'].astype(str)
rs=pa.groupby('year')['paddy_ha'].sum().rename('rs'); yb=pd.read_csv(MID+'/yield_panel_province.csv').set_index('year')['area_ha'].rename('yb')
m=pd.concat([rs,yb],axis=1).dropna()
fig,ax=plt.subplots(1,2,figsize=(DC*0.72,DC*0.32),constrained_layout=True)
sc=ax[0].scatter(m.yb/1e4,m.rs/1e4,c=m.index,cmap='viridis',s=20,edgecolor='white',lw=0.3)
lim=[m.min().min()/1e4*0.9,m.max().max()/1e4*1.05]; ax[0].plot(lim,lim,'--',color='0.5',lw=0.8)
ax[0].set_xlabel('Yearbook sown area (10$^4$ ha)'); ax[0].set_ylabel('Satellite mapped area (10$^4$ ha)')
r=np.corrcoef(m.yb,m.rs)[0,1]; ax[0].set_title('Satellite vs statistical area'); ax[0].text(0.05,0.9,f'r = {r:.2f}, n = {len(m)}\n1:1 line',transform=ax[0].transAxes,fontsize=6.3,va='top')
cb=fig.colorbar(sc,ax=ax[0],shrink=0.8,pad=0.02,ticks=[1985,2000,2020]); cb.set_label('Year',fontsize=6); plab(ax[0],'a')
rd=100*(m.rs-m.yb)/m.yb
ax[1].axhline(0,color='0.6',lw=0.6); ax[1].bar(m.index,rd,color=[OI['orange'] if v>0 else OI['blue'] for v in rd],width=0.8)
ax[1].set_xlabel('Year'); ax[1].set_ylabel('Relative difference (RS−YB)/YB (%)'); ax[1].set_title('Convergence over time')
ax[1].xaxis.set_major_locator(MaxNLocator(5)); ax[1].text(0.04,0.93,'large early divergence\n→ <2% by 2020',transform=ax[1].transAxes,fontsize=6,va='top'); plab(ax[1],'b')
fig.savefig(PUB+'/FigS2_data_agreement.pdf',bbox_inches='tight'); fig.savefig(PUB+'/FigS2_data_agreement.png',dpi=600,bbox_inches='tight'); fig.savefig(PUB+'/FigS2_data_agreement.svg',bbox_inches='tight'); plt.close()
print('FigS2_data_agreement done')
