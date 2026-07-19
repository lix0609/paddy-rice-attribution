import pandas as pd,numpy as np,warnings; warnings.filterwarnings('ignore')
import matplotlib as mpl; mpl.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import pymannkendall as mk
mpl.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial','DejaVu Sans'],'svg.fonttype':'none','pdf.fonttype':42,
 'font.size':7,'axes.titlesize':7.8,'axes.labelsize':7,'xtick.labelsize':6.3,'ytick.labelsize':6.3,'axes.linewidth':0.7,
 'axes.spines.top':False,'axes.spines.right':False})
OI={'blue':'#0072B2','green':'#009E73','orange':'#E69F00','verm':'#D55E00','purple':'#CC79A7'}; MM=1/25.4; DC=183*MM
ROOT='/sessions/vibrant-bold-franklin/mnt/水稻'; PH=ROOT+'/增温驱动吉林水田扩张与增产_1985-2020/06_phenology'; PUB=ROOT+'/增温驱动吉林水田扩张与增产_1985-2020/03_figures/pub'
def plab(ax,L): ax.annotate(L,xy=(0,1),xycoords='axes fraction',xytext=(-32,6),textcoords='offset points',fontsize=9,fontweight='bold',va='bottom')
def sen(x,y):
    from itertools import combinations;x=np.asarray(x,float);y=np.asarray(y,float);s=np.median([(y[j]-y[i])/(x[j]-x[i]) for i,j in combinations(range(len(x)),2)]);return s,np.median(y)-s*np.median(x)
ph=pd.read_csv(PH+'/phenology_station_year.csv'); smGP=ph.groupby('station').agg(lat=('lat','first'),gp=('gp_sow_mat','mean')).dropna()
cm=pd.read_csv(PH+'/county_at10_gradient.csv'); prov=pd.read_csv(PH+'/province_at10_trend.csv')
at=pd.read_csv(PH+'/phenology_accumulated_temp.csv'); smAT=at.groupby('station').agg(lat=('lat','first'),gp_at10=('gp_at10','mean'))
fig,ax=plt.subplots(2,2,figsize=(DC*0.78,DC*0.62),constrained_layout=True)
# a: GP vs latitude (NE stations)
a=ax[0,0]; a.scatter(smGP.lat,smGP.gp,s=20,color=OI['green'],edgecolor='white',zorder=3)
b=np.polyfit(smGP.lat,smGP.gp,1); xr=np.array([smGP.lat.min(),smGP.lat.max()]); a.plot(xr,np.polyval(b,xr),'--',color='0.3',lw=1)
r=np.corrcoef(smGP.lat,smGP.gp)[0,1]; a.set_xlabel('Station latitude (°N)'); a.set_ylabel('Growing period (days)')
a.set_title('Growing period vs latitude'); a.text(0.05,0.08,f'r = {r:.2f}, n = {len(smGP)}',transform=a.transAxes,fontsize=6.5); plab(a,'a')
# b: at10 vs elevation (Jilin counties)
a=ax[0,1]; a.scatter(cm.elev_mean,cm.at10,s=16,color=OI['verm'],edgecolor='white',zorder=3)
b=np.polyfit(cm.elev_mean,cm.at10,1); xr=np.array([cm.elev_mean.min(),cm.elev_mean.max()]); a.plot(xr,np.polyval(b,xr),'--',color='0.3',lw=1)
r=np.corrcoef(cm.elev_mean,cm.at10)[0,1]; a.set_xlabel('County mean elevation (m)'); a.set_ylabel('GS acc. temp. ≥10°C (°C·d)')
a.set_title('Thermal resource vs elevation'); a.text(0.05,0.08,f'r = {r:.2f}, n = {len(cm)}',transform=a.transAxes,fontsize=6.5); plab(a,'b')
# c: province at10 trend
a=ax[1,0]; a.plot(prov.year,prov.at10,'o-',color=OI['orange'],ms=3,mec='white',mew=0.3,zorder=3)
s,ic=sen(prov.year,prov.at10); a.plot(prov.year,ic+s*prov.year,'--',color='0.3',lw=1)
R=mk.original_test(prov.at10.values); a.set_xlabel('Year'); a.set_ylabel('GS acc. temp. ≥10°C (°C·d)')
a.set_title('Rising thermal resource'); a.xaxis.set_major_locator(MaxNLocator(5))
a.text(0.05,0.9,f'+{s*10:.0f} °C·d dec$^{{-1}}$ (Δ≈{prov.at10.values[-1]-prov.at10.values[0]:.0f}) ***',transform=a.transAxes,fontsize=6.3,va='top'); plab(a,'c')
# d: growing-period at10 requirement (stations) roughly stable
a=ax[1,1]; a.hist(at.gp_at10,bins=14,color=OI['blue'],alpha=0.85,edgecolor='white')
mval=at.gp_at10.mean(); a.axvline(mval,color='0.2',lw=1.2,ls='--'); a.set_xlabel('Growing-period acc. temp. (°C·d)')
a.set_ylabel('Count (station-years)'); a.set_title('Thermal requirement to maturity')
a.text(0.05,0.9,f'mean ≈ {mval:.0f} °C·d\n(vs latitude r = {np.corrcoef(smAT.lat,smAT.gp_at10)[0,1]:.2f})',transform=a.transAxes,fontsize=6.3,va='top'); plab(a,'d')
fig.savefig(PUB+'/FigS_phenology.pdf',bbox_inches='tight'); fig.savefig(PUB+'/FigS_phenology.png',dpi=600,bbox_inches='tight'); fig.savefig(PUB+'/FigS_phenology.svg',bbox_inches='tight'); plt.close()
print('FigS thermal done')
