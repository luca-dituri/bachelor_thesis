import numpy as np
from matplotlib.gridspec import GridSpec
import matplotlib.pyplot as plt
from skspatial.objects import Line, Points
import config_gen
import pandas as pd
from itertools import product
from itertools import chain
from tqdm import tqdm
import warnings
import pandas as pd

# Suppress pandas’ SettingWithCopyWarning
pd.options.mode.chained_assignment = None  

# Suppress all FutureWarning (including your Series positional‐indexing warning)
warnings.filterwarnings("ignore", category=FutureWarning)

plt.rcParams['xtick.labelsize'] = 15
plt.rcParams['ytick.labelsize'] = 15
plt.rcParams['axes.labelsize'] = 15


#### sensor_parameters

dist_z = 8.5        # [mm] distance between planes
Lgap = 4.85         # GAP STAVE mm + addition distance for space compliance (mechanincal)
Sgap = 0.15         # GAP CHIP mm
width = 30          # width of the turrets [mm]
stavew=width+Sgap
ProbNoise = config_gen.ProbNoise
Probmiss = config_gen.ProbMiss
mu = config_gen.mu
sigma = config_gen.sigma


def _sample_direction(theta_max_deg=90, uniform_theta=False):
    '''
        Used to sample the direction of generated lines.
    '''

    theta_max = np.deg2rad(theta_max_deg)
    phi = 2*np.pi*np.random.rand()
    if uniform_theta:
        theta = theta_max * np.random.rand()
    else:
        cos_th_min = np.cos(theta_max)
        cos_th = cos_th_min + (1 - cos_th_min) * np.random.rand()
        theta = np.arccos(cos_th)
    d = np.array([np.sin(theta)*np.cos(phi),
                  np.sin(theta)*np.sin(phi),
                  -np.cos(theta)])
    return d / np.linalg.norm(d)

def generate_random_line(point=0, randompoint=True,
                         sensor_xy=(170,150),
                         dz=-17,
                         theta_max_deg=90,
                         uniform_theta=False):
    
    '''
    Generates a random line within the sensor area.
    '''

    if randompoint:
        point0_on_line = np.array([sensor_xy[0]*np.random.rand(),
                                   sensor_xy[1]*np.random.rand(),
                                   0.0])
    else:
        point0_on_line = np.asarray(point, dtype=float)
        if point0_on_line.shape != (3,):
            raise ValueError("point has to be an array of shape (3,)")
    direction = _sample_direction(theta_max_deg=theta_max_deg,
                                  uniform_theta=uniform_theta)
    return point0_on_line, direction


def get_points(vec_reco,cent_reco, zpos,rum=True):

    '''
    Used to get the points that hit my detector, taking them from a given line
    returns a np matrix with three vectors with the coordinates of the hit points
    '''
    x = np.zeros_like(zpos)
    y = np.zeros_like(zpos)
    vec_reco = vec_reco

    for ilay, z in enumerate(zpos):
        x[ilay] = zpos[ilay]*vec_reco[0]/(vec_reco[2]) - (cent_reco[2]*(vec_reco[0]/vec_reco[2])-cent_reco[0])
        y[ilay] = zpos[ilay]*vec_reco[1]/(vec_reco[2]) - (cent_reco[2]*(vec_reco[1]/vec_reco[2])-cent_reco[1])
        
        if (rum==True):
            bias_x, bias_y = apply_noise(x[ilay],y[ilay])
            x[ilay] = bias_x
            y[ilay] = bias_y

    line_points = np.stack([x,y,zpos]).T
    return line_points


def apply_noise(x,y):
    bias_x = x + np.random.normal(mu,sigma)
    bias_y = y + np.random.normal(mu,sigma)
    return bias_x,bias_y


def spatial_limits(points):
    ''' 
        Verifies if all points are within the sensor boundaries.
    '''
    x = points.T[0]
    y = points.T[1]
    return all(x >= 0) and all(x <= 150) and all(y >= 0) and all(y <= 175)

def missed_stave(i, stavew, Lgap):
    ranges = [
        (stavew, stavew + Lgap),
        (2 * stavew + Lgap, 2 * (stavew + Lgap)),
        (stavew + 2 * (stavew + Lgap), 3 * (stavew + Lgap)),
        (stavew + 3 * (stavew + Lgap), 4 * (stavew + Lgap)),
    ]
    return any(start < i < end for start, end in ranges)

class data:
    def __init__( self, indexev, points , direction , pointonline, idnumber ):
        
        self.indexev = indexev #identifies the plot of origin, the event of origin
        self.points = points
        self.direction = direction
        self.pol = pointonline
        self.id = idnumber #identifies if the point is a noise point (-1) or a point of the line (1,2,3)



def noise_gen(ProbNoise, dist_z):
    '''
        Generates noisy pixels in the three planes 
        (maximum of one for plane, maybe add the option of multiple points in every plane)
        return a list of points
    '''
    noise=[]
    j = 0
    for i in range(3):
        if np.random.uniform(0, 1) < ProbNoise:
            x = np.random.uniform(0, 150)
            y = np.random.uniform(0, 30)
            z = -dist_z * i
            if(len(noise) == 0):
                noise = np.array([x, y, z])
            else:
                noise = np.concatenate( [noise,np.array([x, y, z])] , axis = 0 )
    if(len(noise) != 0):
        noise_matrix = noise.reshape(-1,3)
    else:
        noise_matrix = noise
    
    return noise_matrix


def save_csv_py(data,dir):
    ''' 
    A data class is given as an input, and a csv file is created with the following columns:
    event: event number
    x_pos: x coordinate of the hit point
    y_pos: y coordinate of the hit point
    z_pos: z coordinate of the hit point
    trk_index: index of the track (or -1 if noise)
    '''
    x_pos,y_pos,z_pos,trk_id,event=[],[],[],[],[]
    for i in range(len(data)):
        if(len(data[i].points)>0):
            x_pos.append(data[i].points[:,0])
            y_pos.append(data[i].points[:,1])
            z_pos.append(data[i].points[:,2])
            trk_id.append(data[i].id*np.ones_like(data[i].points[:,0]))
            event.append(data[i].indexev*np.ones_like(data[i].points[:,0]))
    x=np.concatenate(x_pos)
    y=np.concatenate(y_pos)
    z=np.concatenate(z_pos)
    id=np.concatenate(trk_id)
    ev=np.concatenate(event)
    d = {"event": ev ,"x_pos": x ,"y_pos": y ,"z_pos": z ,"trk_index": id}
    df = pd.DataFrame(d)
    df['event'] = df['event'].astype(int)
    df['trk_index'] = df['trk_index'].astype(int)
    df.to_csv(dir, index=False)

def read_csv_py(filename):
    df = pd.read_csv(filename)
    x = df['x_pos']
    y = df['y_pos']
    z = df['z_pos']
    ev = df['event'].astype(int)
    trk = df['trk_index'].astype(int)

    return x,y,z,ev,trk

def asSpherical(xyz):
    '''
    Converts unit vectors [x,y,z] to spherical angles [theta, phi] in degrees
    '''
    sph = []
    for x, y, z in xyz:
        theta = np.arccos(z) * 180 / np.pi         # z = cos(theta)
        phi = np.arctan2(y, x) * 180 / np.pi       # atan2 handles full circle
        sph.append([theta, phi])
    return sph

def get_combinations(points, dist_z):
    """
    Returns all valid point combinations across different z-levels.
    """
    levels_z = np.unique(points[:, 2])
    points_by_level = {
        z: np.array([p for p in points if int(p[2]) == int(z) and not is_null(p)])
        for z in levels_z
    }
    valid_levels = [z for z in points_by_level if len(points_by_level[z]) > 0]

    if len(valid_levels) < 2:
        return np.array([])

    if len(valid_levels) == 2:
        return np.array(list(product(points_by_level[valid_levels[0]], points_by_level[valid_levels[1]])))
    else:
        return np.array(list(product(*[points_by_level[z] for z in valid_levels[:3]])))
    
def is_null(point):
    return np.allclose(point, 0)


def data_t_gen(x, y, z, ev, trk):
    """
    Takes data with three spatial coordinates, an event index (generation number),
    and a track index. Returns a list of data objects, each corresponding to a specific track.
    """
    cont = []
    df = pd.DataFrame({'x': x, 'y': y, 'z': z, 'ev': ev, 'trk': trk})
    grouped = df.groupby(['ev', 'trk'])
    
    for (event_id, track_id), group in grouped:
        points = group[['x', 'y', 'z']].values
        if len(points) > 0:
            cont.append(data(event_id, points, 0, 0, track_id))
    return cont


def read_csv_py(filename):
    df = pd.read_csv(filename)
    x = df['x_pos']
    y = df['y_pos']
    z = df['z_pos']
    ev = df['event'].astype(int)
    trk = df['trk_index'].astype(int)

    return x,y,z,ev,trk


def chi2_line(points):

    line = Line.best_fit(points)
    direction = line.direction / np.linalg.norm(line.direction)
    point_on_line = line.point
    diff = points - point_on_line
    cross = np.cross(diff, direction)
    distances = np.linalg.norm(cross, axis=1)
    chi2 = np.sum(distances**2/sigma**2)
    return line,chi2


### HT method



#### fit_parameters

max_point_forfit = 35
n_minpoints_HT = 2
dist_max = 2 # max distance for track association
theta_point_HT = 5000 #5000 # number of points sample with the HT
theta_binning_HT = 500 #500 # number of points sample with the HT
rho_binning_HT = 500 #500 # number of points sample with the HT

#### dumping_tracks
max_res_dump = 1000 # [mm] max error on a track to be dumped

plot_ev_flag = False # save tracker event displays 
plot_ev_thr = 3 # min number of points in the event to be plotted
plot_trk_thr = 0 


def display_single_fit(ev,vec_reco,cent_reco,outname,planelabel):

    eps = 0.0000001

    vec_reco = np.array(vec_reco)
    vec_reco = np.where(vec_reco==0,eps,vec_reco)

    color = ['b','r','g','k','y','m','c','b','r','g','k','y','m','c','b','r','g','k','y','m','c','b','r','g','k','y','m','c','b','r','g','k','y','m','c','b','r','g','k','y','m','c']

    fig = plt.figure(figsize = (20,8))
    gs = GridSpec(2, 3, figure=fig)
    ax1 = fig.add_subplot(gs[0, :2])
    #plt.subplot(1,3,1)
    plt.title('x-z view',fontsize = 15)
    for i,i_trk in enumerate(np.unique(ev.trk_nr.values)):
        ev_trk = ev[ev.trk_nr == i_trk]
        if len(ev_trk)>=2 and i_trk != -1:
            ax1.scatter(ev_trk.x_pos.values,ev_trk.z_pos.values, c = color[int(i_trk)], label = 'track '+str(int(ev_trk.trk_nr.values[0]))+", HT plane "+planelabel, alpha = 0.5, s = 100)
            ax1.plot(np.linspace(0,150),(cent_reco[i][2]-cent_reco[i][0]*vec_reco[i][2]/vec_reco[i][0]) + vec_reco[i][2]/vec_reco[i][0] * np.linspace(0,150), c = color[int(i_trk)], label = 'fit track '+str(int(ev_trk.trk_nr.values[0])))
        else:
            ax1.scatter(ev_trk.x_pos.values,ev_trk.z_pos.values, c = color[-1], label = 'noise', alpha = 0.5, s = 100)
    #ax1.legend()
    ax1.plot([0.,150.],[0.,0.], color = 'blue')
    ax1.plot([0.,150.],[-1*dist_z,-1*dist_z], color = 'blue')
    ax1.plot([0.,150.],[-2*dist_z,-2*dist_z], color = 'blue')
    ax1.set_xlabel('x [mm]')
    ax1.set_ylabel('z [mm]')
    ax1.grid(alpha = 0.5)
    ax1.set_xlim(-10,160)
    ax1.set_ylim(-25.5,8.5)

    #plt.subplot(1,3,2)
    ax2 = fig.add_subplot(gs[1, :2])
    plt.title('y-z view',fontsize = 15)
    for i,i_trk in enumerate(np.unique(ev.trk_nr.values)):
        ev_trk = ev[ev.trk_nr == i_trk]
        if len(ev_trk)>=2 and i_trk != -1:
            ax2.scatter(ev_trk.y_pos.values,ev_trk.z_pos.values, c = color[int(i_trk)], label = 'track '+str(int(ev_trk.trk_nr.values[0]))+", HT plane "+planelabel, alpha = 0.5, s = 100)
            ax2.plot(np.linspace(0.,175.),(cent_reco[i][2]-cent_reco[i][1]*vec_reco[i][2]/vec_reco[i][1]) + vec_reco[i][2]/vec_reco[i][1] * np.linspace(0.,175.), c = color[int(i_trk)], label = 'fit track '+str(int(ev_trk.trk_nr.values[0])))
        else:
            ax2.scatter(ev_trk.y_pos.values,ev_trk.z_pos.values, c = color[-1], label = 'noise', alpha = 0.5, s = 100)
    #ax2.legend()
    for tur in range(0,5):
        shift = tur*(30+Lgap+Sgap)
        ax2.plot([shift + 0.,shift + 30.],[0.,0.], color = 'blue')
        ax2.plot([shift + 0.,shift + 30.],[-1*dist_z,-1*dist_z], color = 'blue')
        ax2.plot([shift + 0.,shift + 30.],[-2*dist_z,-2*dist_z], color = 'blue')
    ax2.set_xlabel('y [mm]')
    ax2.set_ylabel('z [mm]')
    ax2.grid(alpha = 0.5)
    ax2.set_xlim(-10,180)
    ax2.set_ylim(-25.5,8.5)
    
    #plt.subplot(1,3,3)
    ax3 = fig.add_subplot(gs[:, 2])
    plt.title('x-y view',fontsize = 15)
    for i,i_trk in enumerate(np.unique(ev.trk_nr.values)):
        ev_trk = ev[ev.trk_nr == i_trk]
        if len(ev_trk)>=2 and i_trk != -1:
            ax3.scatter(ev_trk.x_pos.values,ev_trk.y_pos.values, c = color[int(i_trk)], label = 'track '+str(int(ev_trk.trk_nr.values[0]))+", HT plane "+planelabel, alpha = 0.5, s = 100)
            ax3.plot(np.linspace(0,150),(cent_reco[i][1]-cent_reco[i][0]*vec_reco[i][1]/vec_reco[i][0]) + vec_reco[i][1]/vec_reco[i][0] * np.linspace(0,150), c = color[int(i_trk)], label = 'fit track '+str(int(ev_trk.trk_nr.values[0])))
        else:
            ax3.scatter(ev_trk.x_pos.values,ev_trk.y_pos.values, c = color[-1], label = 'noise', alpha = 0.5, s = 100)
    #ax3.legend()

    for tur in range(0,5):
        shift = tur*(30+Lgap+Sgap)
        ax3.plot([0.,150.],[shift + 0.,shift + 0.], color = 'blue')
        ax3.plot([0.,150.],[shift + 30.,shift + 30.], color = 'blue')
        ax3.plot([0.,0.],[shift + 0.,shift + 30.], color = 'blue')
        ax3.plot([150.,150.],[shift + 0.,shift + 30.], color = 'blue')
    
    ax3.set_xlabel('x [mm]')
    ax3.set_ylabel('y [mm]')
    ax3.grid(alpha = 0.5)
    ax3.set_xlim(-10,160)
    ax3.set_ylim(-15,180)
    
    plt.tight_layout()
    plt.savefig(outname)
    plt.close()

def res_calculation_Xsigned(ev,vec_reco,cent_reco):
    dist = []
    eps = 0.0000001

    vec_reco = np.array(vec_reco)
    vec_reco = np.where(vec_reco==0,eps,vec_reco)

    for i in range(0,len(ev)):
        x_point = ev.x_pos.values[i]
        x_point_line = ev.z_pos.values[i]/(vec_reco[2]/vec_reco[0]) - (cent_reco[2]-cent_reco[0]*vec_reco[2]/vec_reco[0])/(vec_reco[2]/vec_reco[0])
        dist.append(x_point_line-x_point)
    return dist


def res_calculation_Ysigned(ev,vec_reco,cent_reco):
    dist = []
    eps = 0.0000001

    vec_reco = np.array(vec_reco)
    vec_reco = np.where(vec_reco==0,eps,vec_reco)

    for i in range(0,len(ev)):
        y_point = ev.y_pos.values[i]
        y_point_line = ev.z_pos.values[i]/(vec_reco[2]/vec_reco[1]) - (cent_reco[2]-cent_reco[1]*vec_reco[2]/vec_reco[1])/(vec_reco[2]/vec_reco[1])
        dist.append(y_point_line-y_point)
    return np.array(dist)

def res_calculation(ev,vec_reco,cent_reco):
    dist = []
    for i in range(0,len(ev)):
        point = np.array([ev.x_pos.values[i],ev.y_pos.values[i],ev.z_pos.values[i]])
        point_line = np.array(cent_reco)
        t_min = np.dot(point-point_line,vec_reco)/np.linalg.norm(vec_reco)**2
        dist.append(np.sqrt((cent_reco[0]+t_min*vec_reco[0]-point[0])**2+(cent_reco[1]+t_min*vec_reco[1]-point[1])**2+(cent_reco[2]+t_min*vec_reco[2]-point[2])**2))
    return dist

def distanceSegment3D(p,a,b):
    # normalized tangent vector
    d = np.divide(b - a, np.linalg.norm(b - a))
    # signed parallel distance components
    s = np.dot(a - p, d)
    t = np.dot(p - b, d)
    # clamped parallel distance
    h = np.maximum.reduce([s, t, 0])
    # perpendicular distance component
    c = np.cross(p - a, d)
    return np.hypot(h, np.linalg.norm(c))

def distancePointLine(x_point,y_point,q,m):
    return np.abs(y_point-(m*x_point+q))/(np.sqrt(1+m**2))

def point_Rtheta(x,y,theta):
    return np.cos(theta)*x+np.sin(theta)*y

def fitline(rho,theta):
    x_0 = rho*np.cos(theta)
    y_0 = rho*np.sin(theta)
    q = y_0+x_0/np.tan(theta)
    m = -1/np.tan(theta)
    return q,m

def calc_weight_TrkAss(ev):
    num = len(ev[ev.trk_nr != -1])
    den = len(ev)
    return den-num#10/(num/den)

def IterativeHoughTransform(df_events,i_ev,dim1,dim2):
    #print(dim1,dim2)
    i_track = 0
    df_ev = df_events[df_events.event==i_ev]
    while len(df_ev[df_ev["trk_nr"] == -1]) >= 3:
        #print(df_ev)
        theta_tot, rho_tot = [],[]

        for row in df_ev[df_ev["trk_nr"] == -1].iterrows():

            if (dim1 == "x_pos" and dim2 == "z_pos"):
                theta = np.concatenate((np.linspace(0.,np.pi/2-0.2,int(theta_point_HT/2)),np.linspace(np.pi/2+0.2,np.pi,int(theta_point_HT/2))))
            if (dim1 == "y_pos" and dim2 == "z_pos"):
                theta = np.concatenate((np.linspace(0.,np.pi/2-0.2,int(theta_point_HT/2)),np.linspace(np.pi/2+0.2,np.pi,int(theta_point_HT/2))))
            if (dim1 == "x_pos" and dim2 == "y_pos"):
                theta = np.linspace(0.,np.pi,theta_point_HT)

            rho = point_Rtheta(row[1][dim1],row[1][dim2],theta)
            theta_tot.append(theta)
            rho_tot.append(rho)

        theta_tot = list(chain.from_iterable(theta_tot))
        rho_tot = list(chain.from_iterable(rho_tot))
        hist,bin_x,bin_y = np.histogram2d(theta_tot,rho_tot,bins=[theta_binning_HT,rho_binning_HT], range=([0,np.pi],[-300,300]))
        
        x_ind, y_ind = np.unravel_index(np.argmax(hist), hist.shape)
        bin_x_vec = (bin_x[x_ind]+bin_x[x_ind+1])/2
        bin_y_vec = (bin_y[y_ind]+bin_y[y_ind+1])/2
        q, m = fitline(bin_y_vec,bin_x_vec)
        for row in df_ev[df_ev["trk_nr"] == -1].iterrows():
            x_point, y_point = row[1][dim1], row[1][dim2]
            d = distancePointLine(x_point, y_point, q, m)
            if d < dist_max :
                df_events['trk_nr'].loc[row[0]] = i_track
            #print(d)
        i_track += 1
        df_ev = df_events[df_events.event==i_ev]
        #print(df_ev)
    return df_events




def ChooseBestHough(df_h_xy,df_h_xz,df_h_yz):
    tot_ev = np.unique(np.concatenate((np.unique(df_h_xy.event.values),np.unique(df_h_xz.event.values),np.unique(df_h_yz.event.values)),axis=0))
    df_sel_tot = pd.DataFrame()
    plane_choosen = []
    for __,i_ev in tqdm(enumerate(tot_ev), total = len(tot_ev)):
        skip = False
        plane = ""
        df_sel = pd.DataFrame()
        if len(df_h_xy[df_h_xy.event == i_ev])<=n_minpoints_HT:
            df_sel = df_h_xy[df_h_xy.event == i_ev]
            plane = "xy"
        else:
            fin_res = []
            fin_w = []
            if (len(df_h_xy[(df_h_xy.event == i_ev) & (df_h_xy.trk_nr != -1)])>0):
            
                ev = df_h_xy[(df_h_xy.event == i_ev) & (df_h_xy.trk_nr!=-1)]
                ev_vec_reco, ev_cent_reco, points_onplanes = [], [], []
                res_tot = 0
                for i_trk,j in enumerate(np.unique(ev.trk_nr.values)):
                    trk = ev[ev.trk_nr == j]
                    vec_reco,cent_reco,points_onplanes = ev_fitpoints(trk)
                    res_x = res_calculation_Xsigned(trk,vec_reco,cent_reco)
                    res_y = res_calculation_Ysigned(trk,vec_reco,cent_reco)
                    res_tot += np.sum(np.abs(np.concatenate([res_x,res_y],axis=0)))
                w = calc_weight_TrkAss(df_h_xy[(df_h_xy.event == i_ev)])
                fin_w.append(w)
                fin_res.append(res_tot+w)
                # already good track skip other projections
                #if res_tot<1.:
                #    skip = True
            else:
                fin_res.append(1000)

            if (len(df_h_xz[(df_h_xz.event == i_ev) & (df_h_xz.trk_nr != -1)])>0):

                ev = df_h_xz[(df_h_xz.event == i_ev) & (df_h_xz.trk_nr!=-1)]
                ev_vec_reco, ev_cent_reco, points_onplanes = [], [], []
                res_tot = 0
                for i_trk,j in enumerate(np.unique(ev.trk_nr.values)):
                    trk = ev[ev.trk_nr == j]
                    vec_reco,cent_reco,points_onplanes = ev_fitpoints(trk)
                    res_x = res_calculation_Xsigned(trk,vec_reco,cent_reco)
                    res_y = res_calculation_Ysigned(trk,vec_reco,cent_reco)
                    res_tot += np.sum(np.abs(np.concatenate([res_x,res_y],axis=0)))
                w = calc_weight_TrkAss(df_h_xz[(df_h_xz.event == i_ev)])
                fin_w.append(w)
                fin_res.append(res_tot+w)
                #if res_tot<1.:
                #    skip = True
            else:
                fin_res.append(1000)

            if (len(df_h_yz[(df_h_yz.event == i_ev) & (df_h_yz.trk_nr != -1)])>0):
                
                ev = df_h_yz[(df_h_yz.event == i_ev) & (df_h_yz.trk_nr!=-1)]
                ev_vec_reco, ev_cent_reco, points_onplanes = [], [], []
                res_tot = 0
                for i_trk,j in enumerate(np.unique(ev.trk_nr.values)):
                    trk = ev[ev.trk_nr == j]
                    vec_reco,cent_reco,points_onplanes = ev_fitpoints(trk)
                    res_x = res_calculation_Xsigned(trk,vec_reco,cent_reco)
                    res_y = res_calculation_Ysigned(trk,vec_reco,cent_reco)
                    res_tot += np.sum(np.abs(np.concatenate([res_x,res_y],axis=0)))
                w = calc_weight_TrkAss(df_h_yz[(df_h_yz.event == i_ev)])
                fin_w.append(w)
                fin_res.append(res_tot+w)
                #if res_tot<1.:
                #    skip = True
            else:
                fin_res.append(1000)

            if np.argmin(fin_res) == 0:
                df_sel = df_h_xy[df_h_xy.event == i_ev]
                plane = "xy"
            if np.argmin(fin_res) == 1:
                df_sel = df_h_xz[df_h_xz.event == i_ev]
                plane = "xz"
            if np.argmin(fin_res) == 2:
                df_sel = df_h_yz[df_h_yz.event == i_ev]
                plane = "yz"
        plane_choosen.append(plane)
        df_sel_tot = pd.concat([df_sel_tot,df_sel])
    return df_sel_tot, plane_choosen


def HoughTransform_xy(df_events_old):
    df_events = df_events_old.copy()
    df_events["trk_nr"] = -1*np.ones(len(df_events)) 

    for __,i_ev in tqdm(enumerate(np.unique(df_events.event)),total = len(np.unique(df_events.event))):

        df_ev = df_events[df_events.event==i_ev]
        if len(df_ev)<=n_minpoints_HT:
            for row in df_ev.iterrows():
                df_events['trk_nr'].loc[row[0]] = 0
            continue

        if (len(df_ev)>max_point_forfit):
            continue

        IterativeHoughTransform(df_events,i_ev,'x_pos','y_pos')

    return df_events


def HoughTransform_yz(df_events_old):
    df_events = df_events_old.copy()
    df_events["trk_nr"] = -1*np.ones(len(df_events)) 
    for __,i_ev in tqdm(enumerate(np.unique(df_events.event)),total = len(np.unique(df_events.event))):

        df_ev = df_events[df_events.event==i_ev]

        if len(df_ev)<=n_minpoints_HT:
            for row in df_ev.iterrows():
                df_events['trk_nr'].loc[row[0]] = 0
            continue

        if (len(df_ev)>max_point_forfit):
            continue

        IterativeHoughTransform(df_events,i_ev,'y_pos','z_pos')

    return df_events


def HoughTransform_xz(df_events_old):
    df_events = df_events_old.copy()
    df_events["trk_nr"] = -1*np.ones(len(df_events)) 

    for __,i_ev in tqdm(enumerate(np.unique(df_events.event)),total = len(np.unique(df_events.event))):

        df_ev = df_events[df_events.event==i_ev]

        if len(df_ev)<=n_minpoints_HT:
            for row in df_ev.iterrows():
                df_events['trk_nr'].loc[row[0]] = 0
            continue

        if (len(df_ev)>max_point_forfit):
            continue

        IterativeHoughTransform(df_events,i_ev,'x_pos','z_pos')

    return df_events




def ev_fitpoints(ev):
    if (len(ev) >= 2) and not (np.all(ev.trk_nr.values == -1)):
        points_ =  Points(ev[['x_pos','y_pos','z_pos']].values)
        dir_reco = np.array(Line.best_fit(points_).direction)
        centroid_reco = np.array(Line.best_fit(points_).point)
        if dir_reco[2]>0:
            dir_reco = -dir_reco
        points_onplanes = np.array([[centroid_reco[0]+(0-centroid_reco[2])*dir_reco[0]/dir_reco[2],centroid_reco[1]+(0-centroid_reco[2])*dir_reco[1]/dir_reco[2],0],
                                 [centroid_reco[0]+(-dist_z-centroid_reco[2])*dir_reco[0]/dir_reco[2],centroid_reco[1]+(-dist_z-centroid_reco[2])*dir_reco[1]/dir_reco[2],-dist_z],
                                 [centroid_reco[0]+(-2*dist_z-centroid_reco[2])*dir_reco[0]/dir_reco[2],centroid_reco[1]+(-2*dist_z-centroid_reco[2])*dir_reco[1]/dir_reco[2],-2*dist_z]])
    else:
        dir_reco = [-999,-999,-999]
        centroid_reco = [-999,-999,-999]
        points_onplanes = np.array([[-999,-999,-999],
                                 [-999,-999,-999],
                                 [-999,-999,-999]])
    return dir_reco,centroid_reco,points_onplanes

def fit_all_events(df, label, plane_choosen, out_dir):
    list_dir = []
    list_theta = []
    list_phi = []
    list_ev_number_tracks = []
    list_point = []
    list_points_on_planes = []
    res_1_x, res_1_y, res_2_x, res_2_y, res_3_x, res_3_y = [],[],[],[],[],[]
    list_ev_number_vertex = []
    list_d_vertex = []
    list_coord_vertex = []
    if len(df)>0:

        for i_ev,i in tqdm(enumerate(np.unique(df.event.values)),total=len(np.unique(df.event.values))):
            list_theta_ev = []
            vertex_dir,vertex_point = [],[]
            ev = df[df.event == i]
            ev_vec_reco, ev_cent_reco = [], []
            if np.all(ev.trk_nr.values == -1):
                continue

            for i_trk,j in enumerate(np.unique(ev.trk_nr.values)):

                trk = ev[ev.trk_nr == j]

                vec_reco,cent_reco,points_onplanes = ev_fitpoints(trk)
                vec_z = np.array([0.,0.,-1.])

                res_x = res_calculation_Xsigned(trk,vec_reco,cent_reco)
                res_y = res_calculation_Ysigned(trk,vec_reco,cent_reco)
                res_tot = np.sum(np.abs(np.concatenate([res_x,res_y],axis=0)))

                if np.max(np.abs(res_tot))<max_res_dump and not (np.all(ev[ev.trk_nr == j].trk_nr.values == -1)):
                    list_theta.append(np.arccos(np.clip(np.dot(vec_reco,vec_z),-1,1)))
                    list_theta_ev.append(np.arccos(np.clip(np.dot(vec_reco,vec_z),-1,1)))
                    list_phi.append(np.arctan2(vec_reco[1],vec_reco[0]))
                    list_point.append(cent_reco)   
                    list_points_on_planes.append(points_onplanes) 
                    list_ev_number_tracks.append(i)
                    list_dir.append(vec_reco)
                    vertex_dir.append(vec_reco)
                    vertex_point.append(cent_reco)
                    if len(res_x) == 3:
                        res_1_x.append(res_x[0])
                        res_1_y.append(res_y[0])
                        res_2_x.append(res_x[1])
                        res_2_y.append(res_y[1])
                        res_3_x.append(res_x[2])
                        res_3_y.append(res_y[2])
                ev_vec_reco.append(vec_reco)
                ev_cent_reco.append(cent_reco)

            if len(vertex_point)==2 and np.max(np.abs(res_tot))<max_res_dump and not (np.all(ev[ev.trk_nr == j].trk_nr.values == -1)):
                d_vertex = 10000
                vertex = []
                for t in np.linspace(-4000,4000,1000):
                    a = vertex_point[0]+4000*vertex_dir[0]
                    b = vertex_point[0]-4000*vertex_dir[0]
                    p = vertex_point[1]+t*vertex_dir[1]
                    d_temp = distanceSegment3D(p,a,b)
                    if d_temp <= d_vertex:
                        d_vertex = d_temp
                        vertex = p
                list_ev_number_vertex.append(i)
                list_d_vertex.append(d_vertex)
                list_coord_vertex.append(vertex)

            if plot_ev_flag==True and len(ev)>plot_ev_thr and len(np.unique(ev.trk_nr[ev.trk_nr!=-1].values))>=plot_trk_thr:
                display_single_fit(ev,ev_vec_reco,ev_cent_reco,out_dir+'\\ev_display\\ev_'+str(i)+'_track_'+str(label)+'.png',plane_choosen[i_ev])

    return np.array(list_ev_number_tracks),np.array(list_ev_number_vertex),np.array(list_dir), np.array(list_point), np.array(list_theta), np.array(list_phi), np.array(list_points_on_planes), np.array(res_1_x), np.array(res_1_y), np.array(res_2_x), np.array(res_2_y), np.array(res_3_x), np.array(res_3_y), np.array(list_d_vertex), np.array(list_coord_vertex)


def fit_events(data):

    if len(data)==0:
        return data,[[-999.,-999.,-999.]],[[-999.,-999.,-999.]]
    
    print('Hough Track Seeding running...')
    df_clusters_hough_xy = HoughTransform_xy(data)
    df_clusters_hough_xy['event_trk_nr'] = df_clusters_hough_xy[['event','trk_nr']].apply(lambda x: str(x[0])+'_'+str(x[1]), axis=1)
    horizontal_xy = df_clusters_hough_xy.groupby(by=['event_trk_nr'])['z_pos'].nunique()[df_clusters_hough_xy.groupby(by=['event_trk_nr'])['z_pos'].nunique()<2].index.values
    df_clusters_hough_xy["trk_nr"] = df_clusters_hough_xy[['event_trk_nr','trk_nr']].apply(lambda x: -1 if x[0] in horizontal_xy else x[1], axis=1)

    df_clusters_hough_xz = HoughTransform_xz(data)
    df_clusters_hough_xz['event_trk_nr'] = df_clusters_hough_xz[['event','trk_nr']].apply(lambda x: str(x[0])+'_'+str(x[1]), axis=1)
    horizontal_xz = df_clusters_hough_xz.groupby(by=['event_trk_nr'])['z_pos'].nunique()[df_clusters_hough_xz.groupby(by=['event_trk_nr'])['z_pos'].nunique()<2].index.values
    df_clusters_hough_xz["trk_nr"] = df_clusters_hough_xz[['event_trk_nr','trk_nr']].apply(lambda x: -1 if x[0] in horizontal_xz else x[1], axis=1)

    df_clusters_hough_yz = HoughTransform_yz(data)
    df_clusters_hough_yz['event_trk_nr'] = df_clusters_hough_yz[['event','trk_nr']].apply(lambda x: str(x[0])+'_'+str(x[1]), axis=1)
    horizontal_yz = df_clusters_hough_yz.groupby(by=['event_trk_nr'])['z_pos'].nunique()[df_clusters_hough_yz.groupby(by=['event_trk_nr'])['z_pos'].nunique()<2].index.values
    df_clusters_hough_yz["trk_nr"] = df_clusters_hough_yz[['event_trk_nr','trk_nr']].apply(lambda x: -1 if x[0] in horizontal_yz else x[1], axis=1)

    print('Choose best hough')
    df_clusters_hough_best, plane_choosen = ChooseBestHough(df_clusters_hough_xy,df_clusters_hough_xz,df_clusters_hough_yz)

   # use fit_all_events to get the fit parameters and display
    print('Fit all events')
    ev_number_tracks, ev_number_vertex, vec, point, theta, phi, points_onplanes, res_1_x, res_1_y, res_2_x, res_2_y, res_3_x, res_3_y, d_vertex, coord_vertex = fit_all_events(df_clusters_hough_best,'HT',plane_choosen,"C:\\Users\\lcdit\\OneDrive\\Desktop\\Uni\\Triennale\\Tesi\\metodo2")

    return df_clusters_hough_best,vec,point


def per_event_accuracy_greedy(df, event_col='event', true_col='trk_index', pred_col='trk_nr',
                              unassigned_label=-1, include_unassigned_in_den=True):
    df = df.copy()
    df[true_col] = df[true_col].astype(int)
    df[pred_col] = df[pred_col].astype(int)
    rows = []
    for ev, g in df.groupby(event_col):
        denom = len(g) if include_unassigned_in_den else int((g[pred_col] != unassigned_label).sum())
        if denom == 0:
            rows.append((ev, 0, 0, float('nan')))
            continue
        ct = pd.crosstab(g[true_col], g[pred_col])
        ct_map = ct.drop(columns=[unassigned_label], errors='ignore')
        matched = 0
        if ct_map.size > 0:
            M = ct_map.values
            while M.size and M.max() > 0:
                i, j = np.unravel_index(np.argmax(M), M.shape)
                matched += int(M[i, j])
                M = np.delete(M, i, axis=0)
                M = np.delete(M, j, axis=1)
        rows.append((ev, matched, denom, matched/denom if denom else float('nan')))
    return pd.DataFrame(rows, columns=[event_col, 'correct', 'total', 'accuracy'])