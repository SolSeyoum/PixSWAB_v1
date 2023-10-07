
'''
PiXSWAB model. To run the model, prepare a cinfig file with proper paths and required inputs (eg PixSWAB_config.ini)
and run it in comdaline as python pixswab_v3_cmd.py pixSWAB_config_ini)
'''

# import packages

import os
import numpy as np
import xarray as xr
import rioxarray as rio
import pandas as pd
import datetime
import warnings
import netCDF4
import geopandas as gpd
import sys
import logging
import warnings
import glob
from collections import namedtuple
from configparser import ConfigParser
import ast
import time


# functions 
from dask.distributed import Client, LocalCluster
import multiprocessing as mp
def start_multiprocessing():
    try:
        client = Client('tcp://localhost:8786', timeout='4s')
        return client
    except OSError:
        cluster =  LocalCluster(ip="",n_workers=int(0.9 * mp.cpu_count()),
            scheduler_port=8786,
            processes=False,
            threads_per_worker=4,
            memory_limit='48GB',
        )
    return Client(cluster)
    
def open_nc(nc,chunks):
    with xr.open_dataset(nc, chunks=chunks) as dts:
        key=list(dts.keys())[0]
        var=dts[key]
        dts.close()
        return var,key

def open_nc_not(nc,chunks):
    lonlat = ['longitude','latitude']
    lonlat_chunks = {k: v for k, v in chunks.items() if k in lonlat}
    with xr.open_dataset(nc, chunks=lonlat_chunks) as dts:
        key=list(dts.keys())[0]
        var=dts[key]
        dts.close()
        return var,key


def get_lcc_info(lu, lcc_dict):
    '''
    Converst lcc info from dictionary to 2D array
    '''
    # c_ret = f_consumed = root_depth = lu.copy()
    lug = f_consumed = root_depth = lu.copy()

    for key in lcc_dict.keys():
        lu_code=lcc_dict[key][0]
        # catchment_retension = lcc_dict[key][3]
        # c_ret = c_ret.where(lu!=lu_code,catchment_retension)
        lug_code = lcc_dict[key][3]
        lug = lug.where(lu!=lu_code,lug_code)
        
        consumed_fractions = lcc_dict[key][2]
        f_consumed = f_consumed.where(lu!=lu_code,consumed_fractions)
        
        rd = lcc_dict[key][1]
        root_depth = root_depth.where(lu!=lu_code,rd)
        
    ds = xr.Dataset()
    # ds['c_ret'] = c_ret
    ds['lug'] = lug
    ds['f_consumed'] = f_consumed
    ds['root_depth'] = root_depth
    return ds ## return c_ret, f_consumed, root_depth

def get_lug_stc_info(lug, stc, lug_stc_dict):
    '''
    Converst lcg_stc info from dictionary to 2D array
    '''
    rt = rtc = rtm = rtmf = rtf = rtvf = lug.copy()
    # print(lug_stc_dict.keys())
    for key in lug_stc_dict.keys():
        lug_code=lug_stc_dict[key][0]
        # print(key, lug_code, lug_stc_dict[key][1])
        crc = lug_stc_dict[key][1]
        crm = lug_stc_dict[key][2]
        crmf = lug_stc_dict[key][3]
        crf = lug_stc_dict[key][4]
        crvf = lug_stc_dict[key][5]
        rtc = rtc.where(lug!=lug_code,crc)
        rtm = rtm.where(lug!=lug_code,crm)
        rtmf = rtmf.where(lug!=lug_code,crmf)
        rtv = rtf.where(lug!=lug_code,crf)
        rtvf = rtvf.where(lug!=lug_code,crvf)
    
    # get unique values from stc
    stc_val=stc.values
    stc_unq=np.unique(stc_val[~np.isnan(stc_val)])
    # print(stc_unq)
    for sc in stc_unq:
        rt = rt.where((sc!=1.)&(sc!=5.), rtc)
        rt = rt.where(sc!=2., rtvf)
        rt = rt.where(sc!=3., rtf)
        rt = rt.where(sc!=4., rtmf)
        rt = rt.where(sc!=6., rtm)
    del rtc, rtm, rtmf, rtf, rtvf   
    
    return rt ## return c_ret, f_consumed, root_depth
    

    
def area_sqkm(ds):
    '''
    Calculate pixel area in square kilometers
    '''
    dlat = abs(ds.latitude[-1]-ds.latitude[-2])
    dlon = abs(ds.longitude[-1]-ds.longitude[-2])

    R = 6378137 # Radius of earth  6.37e6

    # we know already that the spacing of the points is one degree latitude
    dϕ = np.deg2rad(dlat)
    dλ = np.deg2rad(dlon)
    dA = R**2 * dϕ * dλ * np.cos(np.deg2rad(ds.latitude))
    # pixel area in square meter
    pixel_area = dA.where(ds.notnull())
    # pixel area in square kilometer
    pixel_area = pixel_area/1e6 ## Area in square kilometers
    ds.close()
    return pixel_area

def log(file, text, level):
    infoLog = logging.FileHandler(file)
    # infoLog.setFormatter(format)
    logger = logging.getLogger(file)
    logger.setLevel(level)
    
    if not logger.handlers:
        logger.addHandler(infoLog)
        if (level == logging.INFO):
            logger.info(text)
        if (level == logging.ERROR):
            logger.error(text)
        if (level == logging.WARNING):
            logger.warning(text)
    
    infoLog.close()
    logger.removeHandler(infoLog)
    
    return

def check_allignment(ds1, ds2, name_ds2):
        try:
            da1,ds2 = xr.align(ds1,ds2, join='exact')   # will raise a ValueError if not aligned
        except:
            print('the {0} does not align with precipitation dataset.'.format(name))
            print('Model exits! Check the data.')
            sys.exit()
        return
   

## Lambda functions
sum2 = lambda a,b: a+b
dif2 = lambda a,b: a-b
def subtract(a,b):
    return xr.apply_ufunc(dif2, a, b, dask = 'allowed') 
def add(a,b):
    return xr.apply_ufunc(sum2, a, b, dask = 'allowed') 


##---------------##
## PixSWAB model ##
##---------------##
def prepare_inputs(iniFile):
    
    parser = ConfigParser()
    parser.read(iniFile)

    # Create output folder with the time and date
    root = parser.get('FILE_PATHS','PathRoot')
    output_dir = os.path.join(root,parser.get('FILE_PATHS', 'PathOut')) 
    time_now = datetime.datetime.now()
    time_str = time_now.strftime('%Y_%m_%d_%Hh_%Mm')
    dir_out = os.path.join(output_dir, str(time_str))
    if not os.path.exists(dir_out):
        os.makedirs(dir_out)

    
    # Create log file
    log_file_path = os.path.join(dir_out, 'log_file.log')
    txt = '{t}: PixSWAB model'.format(t=time_str)
    log(log_file_path, txt, logging.INFO)

    Startdate = datetime.datetime.strptime(parser.get('PERIOD','ForcingStart'),"%d/%m/%Y")  # Start of forcing
    Enddate = datetime.datetime.strptime(parser.get('PERIOD','ForcingEnd'),"%d/%m/%Y") 

    format_log = logging.Formatter('%(message)s :%(message)s\n')
    log(log_file_path, '{:>26s}: {}'.format('Start date', str(Startdate)), logging.INFO)
    log(log_file_path, '{:>26s}: {}'.format('End date', str(Enddate)), logging.INFO)
    
    # Read ctachmnet characterstics files (lcc file, runoffcoefficnet file etc)
    lcc_info_file = os.path.join(root,parser.get('PARAMETER_LOOKUP', 'lccInfo'))
    rc_file = os.path.join(root,parser.get('PARAMETER_LOOKUP', 'potRC'))
    rc_slope_file = os.path.join(root,parser.get('PARAMETER_LOOKUP', 'potRCSlope'))
    # Read the csv files
    df=pd.read_csv(lcc_info_file,sep=',',index_col=0)
    lcc_dict = df.T.to_dict('list')

    df2=pd.read_csv(rc_file,sep=',',index_col=0)
    lug_stc_dict = df2.T.to_dict('list')

    df3=pd.read_csv(rc_slope_file,sep=',',index_col=0)
    lug_stc_slope_dict = df3.T.to_dict('list')

    # print(df)
    log(log_file_path, '{:>26s}:'.format('Input files and parameters'), logging.INFO)
    formater = '{:>26s}: {}'
    log(log_file_path, formater.format('LCC info file path', str(lcc_info_file)), logging.INFO)
    log(log_file_path, formater.format('rc info file path', str(rc_file)), logging.INFO)
    log(log_file_path, formater.format('RC slope info file path', str(rc_slope_file)), logging.INFO)

    # prepare input information in a dict
    period = {
             's': Startdate,
             'e': Enddate
             }

    input_files =  {
                    'p_in' : os.path.join(root, parser.get('INPUT_NC_FILES', 'pcp')),
                    'e_in' : os.path.join(root, parser.get('INPUT_NC_FILES', 'et')),
                    'i_in' : os.path.join(root, parser.get('INPUT_NC_FILES', 'intp')),
                    'lu_in' : os.path.join(root, parser.get('INPUT_NC_FILES', 'lcc')),
                    'thetasat' : os.path.join(root, parser.get('INPUT_NC_FILES', 'soilSat')),
                    'slope': os.path.join(root, parser.get('INPUT_NC_FILES', 'bslope')),
                    'stc': os.path.join(root, parser.get('INPUT_NC_FILES', 'soilTxt')),
                    'sfc': os.path.join(root, parser.get('INPUT_NC_FILES', 'soilFC')),
                    }
    
    pixswab_parameters =  {   
                    'baseFlowReces':  float(parser.get('MODEL_PARAMS', 'baseFlowReces')),
                    'deepPerc':float(parser.get('MODEL_PARAMS', 'deepPerc')),
                    'rcExp':float(parser.get('MODEL_PARAMS', 'rcExp')),
                  }

    # Specify the output netCDF files based on your need
    # full list of possible outputs: ['ETg', 'ETb', 'SRO', 'Qbf', 'SM', 'Perc']
    # output = ['ETg', 'ETb', 'SRO', 'Qbf', 'SM', 'Perc', 'GW', 'Qs_unmet', 'dPerc']
    output = ast.literal_eval(parser.get('MODEL_OUTPUTS', 'reqOutputList'))

    # encoding for writing dataset to file
    #chunk = -1
    latchunk = int(parser.get('NETCDF_ATTRIBUTES', 'latchunk'))
    lonchunk = int(parser.get('NETCDF_ATTRIBUTES', 'lonchunk'))
    timechunk = int(parser.get('NETCDF_ATTRIBUTES', 'timechunk'))
    chunks = {'time':timechunk,'latitude':latchunk, 'longitude':lonchunk}

    # read the sizes of the precipitation dataset
    xds=xr.open_dataset(input_files['p_in'])

    latchunk = min(latchunk,xds.latitude.size)
    lonchunk = min(lonchunk,xds.longitude.size)
    timechunk = 1
    chunks = {'time':timechunk,'latitude':latchunk, 'longitude':lonchunk}

    # chunks = [timechunk,latchunk, lonchunk]
    comp = dict(zlib=True, complevel=9, least_significant_digit=2, chunksizes=list(chunks.values()))  

    encoding_output = dict(
                    dtype =  np.float32,
                    _FillValue = np.nan,
                    scale_factor = np.float32(1.0),
                    add_offset = np.float32(0.0),
                    zlib = True,
                    complevel = 9,
                    least_significant_digit = 2,
                    chunksizes=tuple(chunks.values())
                )

    xds.close()

    pixswab_inputs = {
        'period':period,
        'input_files':input_files,
        # 'parameters':parameters,
        'chunks':chunks,
        'output':output,
        'lcc_dict':lcc_dict,
        'lug_stc_dict':lug_stc_dict,
        'lug_stc_slope_dict':lug_stc_slope_dict
      }

    for key, value in input_files.items():
        log(log_file_path, '{:>26s}: {}'.format(key, str(value)), logging.INFO)
    for key, value in pixswab_parameters.items():
        log(log_file_path, '{:>26s}: {}'.format(key, str(value)), logging.INFO)

    return pixswab_inputs, pixswab_parameters, log_file_path, dir_out, encoding_output

def get_input(pixswab_inputs):
    '''
    Runs PixSWAB model. pixswab_params is a dictionary with the following keys:
    
    'period': dict() 
        contains keys: 's' and 'e' for startdate and enddate of the simulation 
        with format 'YYYY-MM-DD'
    'input_files': dict()
        contains keys: 'p_in', 'e_in', 'lu_in', 'I_in', 'thetasat', 'slope', 'stc' and 'sfc' 
        Each dictionary value is a string which contains the path to the netCDF 
        file containing the input data for precipitation,
        evapotranspiration, landuse, interception, saturated water content, slope, soil tecture class  
        and soil filed capacity
   'chunks': dict()
        contains keys: 'time', 'latitude', 'longitude'
        Each dictionary value contains the size of the chunk in the mentioned 
        dimensions for xarray (int)
   'output':list()
        List contains the names of the variables to be printed to output. 
        Potential outputs are: ['ETg', 'ETb', 'SRO', 'Qbf', 'SM', 'Perc', 'GW', 'Qs_unmet', 'dPerc']
   'lcc_dict': dict()
        contains one key per land cover class. Dictionary values are lists 
        containing land cover parameters:
        lcc code,root depth,consumed fraction,potential runoff coefficnet and landuse group
            example: 'Rainfed Cropland': [41, 510.0, 1.0, 3]
    'lug_stc_dict': dict()
        contains one key per land use. Dictionary values are lists 
        containing ptential runoff  coefficient parameters for landuse groups per different soil tecture 
        calsess for float slope (zero slope):
        Land use, code, Coarse, Medium, Medium fine, Fine, Very fine
        example: 'Forest' [1, 0.05, 0.1, 0.17, 0.3, 0.4]
    'lug_stc_slope_dict': dict()
        contains one key per land use. Dictionary values are lists 
        containing ptential coefficent of a linear equation to calculate potential runoff coefficnet for 
        different slopes er different soil tecture calsess
        Land use, code, Coarse, Medium, Medium fine, Fine, Very fine
         example: 'Grass' [2, 0.57, 0.522, 0.46, 0.35, 0.26]
    '''
        
    inputs = dict()
    # Get period of the simulation
    period = pixswab_inputs['period']
    inputs['period']=period
    
    # print('Start period', period['s'], '\nEnd period', period['e'])
    # y = datetime.datetime.strptime(period['s'], '%Y-%m-%d').year # starting year of the simulation
    y = period['s'].year # starting year of the simulation

    # Get list of input files, parameters and chunks size
    input_files = pixswab_inputs['input_files']
    # parameters = pixswab_inputs['parameters']
    chunks = pixswab_inputs['chunks']

    # read the netCDF files
    Pt,_= open_nc(input_files['p_in'],chunks)
    E,_= open_nc(input_files['e_in'],chunks)
    I,_= open_nc_not(input_files['i_in'],chunks)
    LU,_= open_nc(input_files['lu_in'],chunks)
    thetasat,_= open_nc_not(input_files['thetasat'],chunks)
    stc,_= open_nc_not(input_files['stc'],chunks)
    sfc,_= open_nc_not(input_files['sfc'],chunks)
    
     # read slope
    slope,_= open_nc_not(input_files['slope'],chunks)
    
    # extract the dataset for the start and end time
    Pt = Pt.sel(time=slice(period['s'],period['e']))
    E = E.sel(time=slice(period['s'],period['e']))
    I = I.sel(time=slice(period['s'],period['e']))
    
    ## check alignments of the datasets
    check_allignment(Pt, E, 'evapotranspiration')
    check_allignment(Pt, I, 'interception')
    check_allignment(Pt[0], LU[0], 'landcovercalss')
    check_allignment(Pt[0], slope, 'slope')
    check_allignment(Pt[0], thetasat, 'soil saturated content')
    check_allignment(Pt[0], stc, 'soil texture class')
    check_allignment(Pt[0], sfc, 'soil feild capacity')   
              
    
    
    if((LU.time[0]<=np.datetime64(period['s'])) & (LU.time[-1]>=np.datetime64(period['e']))):
        LU = LU.sel(time=slice(period['s'],period['e']))
    
    # get the lcc information
    lcc_dict = pixswab_inputs['lcc_dict']
    lug_stc_dict = pixswab_inputs['lug_stc_dict']
    lug_stc_slope_dict = pixswab_inputs['lug_stc_slope_dict']
    
    # list of output requested
    output_list = pixswab_inputs['output'] 
    
    inputs['pcp'] = Pt
    inputs['et'] = E
    inputs['intc'] = I
    inputs['LU'] = LU
    inputs['sat_SM'] = thetasat
    inputs['stc'] = stc
    inputs['sfc'] = sfc
    inputs['slope'] = slope
    inputs['lcc'] = lcc_dict
    inputs['lug_stc'] = lug_stc_dict
    inputs['lug_stc_slope'] = lug_stc_slope_dict
    inputs['output_list'] = output_list
 
    return inputs
def initialize(Pt):
    
    # Initialize some variables
    zar = Pt[0]*0  # zero array with the dimension of the inputs
    ar1 = 1+zar  ## arrays of one
    tol = zar+0.999
    mintol = 0.000001
    SM = zar
    GW = zar  
        
    # Select years from monthly time index
    tt=pd.to_datetime(Pt.time.values)
    ss= tt.year
    seen = set()
    seen_add = seen.add
    years = [x for x in ss if not (x in seen or seen_add(x))]
    
    return zar, ar1, tol, mintol, SM, GW, years

def run_model(inputs, pixswab_parameters):
     
    Pt = inputs['pcp'] 
    E = inputs['et']
    I  = inputs['intc']
    LU = inputs['LU'] 
    thetasat = inputs['sat_SM'] 
    stc = inputs['stc']  
    sfc =  inputs['sfc'] 
    slope = inputs['slope'] 
    lcc_dict = inputs['lcc'] 
    lug_stc_dict = inputs['lug_stc'] 
    lug_stc_slope_dict =  inputs['lug_stc_slope']
    
    zar, ar1, tol, mintol, SM, GW, years = initialize(Pt)
    
     # Gridize the parameters
    d_bf = pixswab_parameters['baseFlowReces']*ar1
    dp = pixswab_parameters['deepPerc']*ar1
    rcaf = pixswab_parameters['rcExp']  
    
    for j in range(len(years)):
 
        if (j==0):
            mn_len_bf = 0
            months=Pt.time.where(Pt.time.dt.year.isin(years[j]),drop=True)
        else:
            mn_len_bf = len(Pt.time.where(Pt.time.dt.year.isin(years[:j]),drop=True))
            months=Pt.time.where(Pt.time.dt.year.isin(years[j]),drop=True)

        # print(mn_len_bf, len(months))
        if(len(LU.time)==0):
           lu = LU.isel(time=0)
        elif(j < len(LU.time)):
           lu = LU.isel(time=j)
        else:
           lu = LU.isel(time = len(LU.time)-1)
        
        #mask lu for water bodies
        mask = xr.where((lu==80)|(lu==81)|(lu==90), 1,0) # #include flooded shrub?
        
        ds_par = get_lcc_info(lu, lcc_dict)
        lug = ds_par['lug']
        f_consumed = ds_par['f_consumed']
        Rd = ds_par['root_depth']
        
        # get the potential runoff coefficient 
        rc0 = get_lug_stc_info(lug, stc, lug_stc_dict)
        rc_slope = get_lug_stc_info(lug, stc, lug_stc_slope_dict)
        rc = rc0+(1-rc0)*(slope/(rc_slope+slope))
        rc = rc.where(rc<tol, tol)# Makesure the rc is not greater than 1
        del rc0,rc_slope
        # rc.plot()
        # Calculate saturated soil mositure
        SMmax = thetasat*Rd
        SMfc = sfc*Rd # soil mositure in the root zone
        
        for i in range(len(months)):          
            t = i+mn_len_bf
            # print('\rmonth: ', t+1, ' of', len(Pt.time), end='')
 
            P = Pt.isel(time=t)
            Int = I.isel(time=t)
            ETa = E.isel(time=t)
            SMt_1 = SM #.where(SM>mintol, zar) 
            GWt_1 = GW #.where(GW>mintol, zar) 
            
            ETb,ETg,SRO,Qsupply,SM,perc,Qb,Qs_unmet,GW,dperc = compute_timestep(P,Int,ETa,
                                                                              SMmax,SMfc,SMt_1, GWt_1,rc,rcaf,
                                                                              d_bf, dp,f_consumed,mask,zar, 
                                                                              tol, mintol)
            
            if t == 0:
                etb = ETb
                etg = ETg
                sro = SRO
                qsup = Qsupply
                sm = SM
                prc = perc
                Qbf = Qb
                qsup_unmet = Qs_unmet
                gw = GW
                dprc = dperc
            else:
                etb = xr.concat([etb, ETb], dim='time')  
                etg = xr.concat([etg, ETg], dim='time')
                sro = xr.concat([sro, SRO], dim='time')
                qsup =  xr.concat([qsup, Qsupply], dim='time')
                sm =  xr.concat([sm, SM], dim='time')
                prc =  xr.concat([prc, perc], dim='time')
                Qbf =  xr.concat([Qbf, Qb], dim='time')
                qsup_unmet =  xr.concat([qsup_unmet, Qs_unmet], dim='time')
                gw =  xr.concat([gw, GW], dim='time')
                dprc =  xr.concat([dprc, dperc], dim='time')
            
    mod_results = dict()
    mod_results['etg'] = etg
    mod_results['etb'] = etb
    mod_results['sro'] = sro
    mod_results['qsup'] = qsup
    mod_results['sm'] = sm
    mod_results['prc'] = prc
    mod_results['Qbf'] = Qbf
    mod_results['qsup_unmet'] = qsup_unmet
    mod_results['gw'] = gw
    mod_results['dprc'] = dprc
    
    return mod_results



def compute_timestep(P,Int,ETa,SMmax,SMfc, SMt_1, GWt_1, rc, rcaf, d_bf, dp, f_consumed,mask,zar, tol, mintol):
   
    warnings.filterwarnings("ignore", message='invalid value encountered in greater')
    warnings.filterwarnings("ignore", message='divide by zero encountered in true_divide')
    warnings.filterwarnings("ignore", message='invalid value encountered in true_divide')
    warnings.filterwarnings("ignore", message='overflow encountered in exp')
    warnings.filterwarnings("ignore", message='overflow encountered in power')
    warnings.filterwarnings("ignore", message='invalid value encountered in subtract')
    warnings.filterwarnings("ignore", message='All-NaN slice encountered')
    warnings.filterwarnings("ignore", message='xarray.ufuncs is deprecated')
    
    # SM_ratio = (SMt_1-SMfc)/(SMmax-SMfc)
    SM_ratio = (SMt_1)/(SMmax)
    SM_ratio = SM_ratio.where(SM_ratio>0, 0)

    # adjuste the RC based on the soil moisture (antecedent soil moisture?)
    rc = rc*(SM_ratio**rcaf)
    ETb = ETg = SRO = Qsupply = perc = Qb = SM = GW = zar
    Int = Int.where(Int<=P, P)
    PminInt = subtract(P, Int)
    SRO = PminInt*rc
    Peff = subtract(PminInt, SRO) 

    SMtemp = add(SMt_1, Peff)
    ETaminInt = subtract(ETa,Int)
    ETg = add(ETaminInt.where(SMtemp>=ETaminInt, SMtemp),Int)
    ETb = subtract(ETa,ETg)

    # Update SMtemp
    SMtemp = subtract(SMtemp, ETaminInt)

    #if ETblue can be satisfied from SRO, it should be considered as ETg
    ETg = add(ETg,ETb).where(ETb<=SRO,add(ETg, SRO))
     # Update SRO
    SRO = subtract(SRO, ETb).where(ETb<=SRO,zar)

    # update ETb
    ETb = subtract(ETa,ETg)
    # compute supply to ssatisfy ETblue                                
    Qsupply = (ETb/f_consumed).where(f_consumed>zar, ETb)


    # the diiference b/n Qsupply and ETincr- where does it belong? (runoff, SM, ..?)
    # This should happen only for irrigated crops where fconsumed is less than 1. Otherwise it comes               # from GW
    #add it to SM and if SM is saturated, the remianing is added to SRO
    extra_Qsupply = subtract(Qsupply, ETb)
    extra_Qsupply = extra_Qsupply.where(extra_Qsupply>mintol, zar)
    GW_supply = subtract(Qsupply, extra_Qsupply)

    SMtemp = add(SMtemp, Qsupply) 
    # SMtemp = SMtemp.where(SMtemp<SMmax, SMmax)
    SRO_incr = subtract(SMtemp, SMmax)
    SRO_incr = SRO_incr.where(SRO_incr>mintol, zar)
    # update SRO
    SRO = xr.apply_ufunc(sum2, SRO, SRO_incr, dask = 'allowed') 

    #if SRO from water bodies shoukd not contribute to discharge
    SRO = zar.where((mask==1), SRO)

    SMtemp = SMtemp.where(SMtemp<SMmax, SMmax)

    # percolation happens if SM is greater than FC
    perc = subtract(SMtemp, SMfc)
    perc = perc.where(perc>mintol, zar)

    # Update soil moisture
    SM = SMtemp-perc
    SM = SM.where(SM>mintol, zar)
    SM = SM.where(SM<SMmax, SMmax)

#           ### Calculate ETincr, ETrain, Qsupply, and update SM

    # Calculate baseflow Qb:
    Qb = d_bf*GWt_1
    Qb = Qb.where(Qb>mintol, zar)

    # calculate deep percolation (lost from GW)
    dperc = (GWt_1-Qb)*dp

    # update groundwater storage 
    GW = GWt_1 - Qb - dperc + perc   
    GW = GW.where(GW>mintol, zar)

    # the Qsupply form non-irrigated part is assumed supplied from GW
    # update GW 
    GW = subtract(GW, GW_supply)
    # if GW is negative, it is considered as unmeet supply?
    Qs_unmet = (-1*GW).where(GW<0, zar)
    GW = GW.where(GW>mintol, zar)

    ETb['time'] = P.time
    ETg['time'] = P.time
    SRO['time'] = P.time
    Qsupply['time'] = P.time
    SM['time'] = P.time
    perc['time'] = P.time
    Qb['time'] = P.time
    Qs_unmet['time'] = P.time
    GW['time'] = P.time
    dperc['time'] = P.time

    return ETb,ETg,SRO,Qsupply,SM,perc,Qb,Qs_unmet,GW,dperc

# def prepare_output(output_list, etb, etg, sro, qsup, sm, prc, Qbf, qsup_unmet, gw, dprc):
def prepare_output(output_list, model_outputs):
    
    results = namedtuple("results", ['ET_green', 'ET_blue', 'Surface_runoff', 
                                     'Base_flow','Extra_demand_for_ET_blue',
                                     'Soil_moisture', 'Percolation', 'Groundwater_storage', 
                                     'Unmet_demand_for_ET_blue', 'Deep_percolation'])
    
    if('ET_green' in output_list):
        attrs={"units":"mm/month", "source": "WA+ analysis", "quantity":"ET_green"}
        etg = model_outputs['etg']
        etg.attrs=attrs
        etg.name = 'ET_green'
    else:
        etg = None
    if('ET_blue' in output_list):
        attrs={"units":"mm/month", "source": "WA+ analysis", "quantity":"ET_blue"}
        etb = model_outputs['etb']
        etb.attrs=attrs
        etb.name = 'ET_blue'   
    else:
        etb = None
    
    if('Surface_runoff' in output_list):
        attrs={"units":"mm/month", "source": "WA+ analysis", "quantity":"Surface_runoff"}
        sro = model_outputs['sro']
        sro.attrs=attrs
        sro.name = 'Surface_runoff'
    else:
        sro = None
    
    if('Base_flow' in output_list):
        attrs={"units":"mm/month", "source": "WA+ analysis", "quantity":"Base_flow"}
        Qbf = model_outputs['Qbf']
        Qbf.attrs=attrs
        Qbf.name = 'Base_flow'
    else:
        Qbf = None

    if('Soil_moisture' in output_list):
        attrs={"units":"mm/month", "source": "WA+ analysis", "quantity":"Soil_moisture"}
        sm = model_outputs['sm']
        sm.attrs=attrs
        sm.name = 'Soil_moisture'
        
    else:
        sm = None

    if('Extra_demand_for_ET_blue' in output_list):
        attrs={"units":"mm/month", "source": "WA+ analysis", "quantity":"Extra_demand_for_ET_blue"}
        qsup = model_outputs['qsup']
        qsup.attrs=attrs
        qsup.name = 'Extra_demand_for_ET_blue'
    else:
        qsup = None
        
    if('Percolation' in output_list):
        attrs={"units":"mm/month", "source": "WA+ analysis", "quantity":"Percolation"}
        prc = model_outputs['prc']
        prc.attrs=attrs
        prc.name = 'Percolation'
    else:
        prc = None
        
    if('Groundwater_storage' in output_list):
        attrs={"units":"mm/month", "source": "WA+ analysis", "quantity":"Groundwater_storage"}
        gw = model_outputs['gw']
        gw.attrs=attrs
        gw.name = 'Groundwater_storage'
    else:
        gw = None
    
    if('Unmet_demand_for_ET_blue' in output_list):
        attrs={"units":"mm/month", "source": "WA+ analysis", "quantity":"Unmet_demand_for_ET_blue"}
        qsup_unmet = model_outputs['qsup_unmet']
        qsup_unmet.attrs=attrs
        qsup_unmet.name = 'Unmet_demand_for_ET_blue'
    else:
        qsup_unmet = None
    
    if('Deep_percolation' in output_list):
        attrs={"units":"mm/month", "source": "WA+ analysis", "quantity":"Deep_percolation"}
        dprc = model_outputs['dprc']
        dprc.attrs=attrs
        dprc.name = 'Deep_percolation'
    else:
        dprc = None

    return results(etg, etb, sro, Qbf, qsup, sm, prc, gw, qsup_unmet, dprc)

def run_pixswab(pixswab_inputs, pixswab_parameters):
    inputs = get_input(pixswab_inputs)
    model_outputs = run_model(inputs, pixswab_parameters)
    return prepare_output(pixswab_inputs['output'],model_outputs)  

def write_output_files(result, log_file_path, dir_out, encoding_output):
    # from create_nc_funcs import write_to_netcdf
    # from dask.distributed import progress
    # import time
    t1 = time.perf_counter ()

    log(log_file_path, '\n{:>26s}:'.format('Output files path'), logging.INFO)

    # print(f"\nwriting the netcdf files")   
    for r in result:
        if(r is not None):
            # print(r)
            # print("* {0}".format(r.name))
            nc_fn=r.name+'.nc'
            nc_path=os.path.join(dir_out,nc_fn)
            encoding = {r.name: encoding_output}
            r.to_netcdf(nc_path,encoding=encoding)
    #         r.close()
            # delayed_obj = r.to_netcdf(nc_path,encoding=encoding, compute=True)
            # result = delayed_obj#.persist()
            # progress(result)
            r.close()
            name = r.name
            log(log_file_path, '{:>26s}: {}'.format(name, str(nc_path)), logging.INFO)
            # del r
    # print(f"Writing the netCDFs is completed!")
    t2 = time.perf_counter ()
    time_diff = t2 - t1
    # print(f"\nIt took {time_diff} Secs to execute this method")

    time_now = datetime.datetime.now()
    time_str = time_now.strftime('%Y-%m-%d %H:%M:%S.%f')
    log(log_file_path, '{t}: PixSWAB model finished.'.format(t=time_str), logging.INFO)
    

def convert_result_to_volume(dss, basin_shp):
    
    
    
    ts_all=[]
    
    # print('Calculating monthly fluxes volume')
    shape = gpd.read_file(basin_shp,crs="EPSG:4326")
    shape = shape.to_crs("EPSG:4326")
    for var in dss: #insert paths to your BaseFlow and SRO output netCDF files
        
        #Clip the datset by shape of the subbasins 
        
        var = var.rio.set_crs("epsg:4326", inplace=True)
        var_clipped = var.rio.clip(shape.geometry.values, shape.crs, drop=False)
        
        # Calculate area of the subbasins
        area = area_sqkm(var_clipped[0])

        # compute volume by multiplying the depth by area
        Volume=var_clipped*area*1e-6
        #Volume=var_clipped

        Volume = Volume.rename('{0}'.format(var.attrs['quantity']))
        # sum the volume spatially and convert it dataframe to have timeseries values
        ts = Volume.sum(dim=['latitude','longitude'], skipna=True).to_dataframe()
        
        del ts['spatial_ref']
        del Volume
        del var
        var_clipped = var_clipped.rename('{0}'.format(var_clipped.attrs['quantity']))
        ts.columns = [var_clipped.attrs['quantity']]
        del var_clipped
        # var.close()
        ts_all.append(ts)
    df = pd.concat(ts_all, axis =1)
    return df


if __name__ == "__main__":
    t = time.time()
    # multiprocessing client
    # client = start_multiprocessing()
    # client.restart()
    
    iniFile = os.path.normpath(sys.argv[1])
    pixswab_inputs, pixswab_parameters, log_file_path, out_dir, encoding_output = prepare_inputs(iniFile)
    result = run_pixswab(pixswab_inputs, pixswab_parameters)
    write_output_files(result, log_file_path, out_dir, encoding_output)
    
    elapsed = time.time() - t
    print(">> Time elapsed: "+"{0:.2f}".format(elapsed)+" s")