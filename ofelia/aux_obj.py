import os
import numpy as np

from pyXSteam.XSteam import XSteam
from scipy.stats import linregress as LR

from dolfinx.fem import (assemble_scalar, form)
import ufl
from ufl import inner, grad
import openmc

from mpi4py import MPI

#save to file lib
from dolfinx.io import XDMFFile

#for extracting data along a line (extract cells function)
import dolfinx 
from dolfinx import geometry

# Update material files
class updateXML():
    '''
    updateXML is a class

    It is used to update materials.xml file. 
    Note: it uses steamTables (XSteam) for computing water density! 

    attributes:
        mat_dict: dictionary of openmc materials (dictionary)
        n_div: number of subdivisions (int)
        pressure: fluid pressure (float)
        Tmin: min value for temperature array (float)
        Tmax: max value for temperature array (float)

    functions:
        update: function that updates the materials xml file with new temperatures
    '''

    def __init__(self, mat_dict: dict, n_div: int, pressure : float = 155., Tmin : float = 280, Tmax : float = 330):
        self.mat_dict = mat_dict
        self.n_div = n_div
        
        steamTable = XSteam(XSteam.UNIT_SYSTEM_MKS)

        Temperature = np.linspace(Tmin, Tmax, 50) + 273.15 # Kelvin
        density = np.zeros_like(Temperature)

        for ii in range(len(density)):
            density[ii] = steamTable.rho_pt(pressure, Temperature[ii]-273.15) / 1e3 # g/cm3

        self.fitting = LR(Temperature, density)
            

    def update(self, Tf = None, Tc = None, sn = None, index = None):

        if Tf is not None:
            for k in range(self.n_div):
                # print(Tf[k])
                self.mat_dict['fuel'][k].temperature = Tf[k]

        if Tc is not None:
            for k in range(self.n_div):
                # print(Tc[k])
                rho = self.fitting.intercept + self.fitting.slope*Tc[k]
                self.mat_dict['coolant'][k].temperature = Tc[k]
                self.mat_dict['coolant'][k].set_density('g/cm3', rho)
        ######################### da togliere
        self.mat_dict['non_updated'][-3].temperature = np.mean(Tf)

        # Collect the materials together and export to XML
        mat_list = list()
        for key in list(self.mat_dict.keys()):
            for mat in self.mat_dict[key]:
                mat_list.append(mat)
        materials = openmc.Materials(mat_list)
        materials.export_to_xml()

        #Create a new folder, copying the it0 and moving the new 'materials.xml' 
        path = os.getcwd()
        folder_0 = path + '/build_xml/it0'
        new_folder = path + '/build_xml/it' + str(index)
        os.system('cp -r ' + folder_0 + ' ' + new_folder)
        os.system('mv materials.xml ' + new_folder + '/materials.xml')
        UpdateParticles(new_folder, sn)


############################################################################################################

# This class is used in FEniCSx to compute norms, integrals and inner products
class norms():
  '''
  norms is a class

  It is used by FEniCSx to compute norms, integrals and inner products

  attributes:
    funSpace: is a FEniCSx function space
    domain: is FEM domain

  functions:
    L2norm: get a field (u) and computes norm L2, returns a scalar
    H1norm: get a field (u) and computes norm H1, returns a scalar
    L2innerProd: get two fields (u and v) and compute the inner product, returns a scalar
    Linftynorm: get a field (u) and compute norm L_infinity, returns a scalar
  '''
  def __init__(self, funSpace, domain):
    self.domain = domain
    self.trial = ufl.TrialFunction(funSpace)
    self.test  = ufl.TestFunction(funSpace)

    self.dx = ufl.Measure('dx', domain=self.domain)
    self.L2inner = inner(self.trial, self.test) * self.dx
    self.H1inner = inner(grad(self.trial), grad(self.test)) * self.dx

  def L2norm(self, u):
    repl_form = form(ufl.replace(self.L2inner, {self.trial: u, self.test: u}))
    return np.sqrt( self.domain.comm.allreduce(assemble_scalar(repl_form), op=MPI.SUM) )
    
  def H1norm(self, u):
    repl_form = form(ufl.replace(self.H1inner, {self.trial: u, self.test: u}))
    return np.sqrt( self.domain.comm.allreduce(assemble_scalar(repl_form), op=MPI.SUM) )
    
  def L2innerProd(self, u, v):
    repl_form = form(ufl.replace(self.L2inner, {self.trial: u, self.test: v}))
    return self.domain.comm.allreduce(assemble_scalar(repl_form), op=MPI.SUM)

  def Linftynorm(self, u):
    return self.domain.comm.allreduce(np.max(np.abs(u.x.array)), op = MPI.MAX)
   
# Evaluate the normalization of the quantity 'q', knowing the Power 'P' and the pin-lenght 'l' with radius 'r'
class extract_power():
    '''
    extract_power is a class

    attributes:
        n_div: number of pin subdivisions
        power: system thermal power
        mesh_size: TBD
        length: length of the pin (unit of measure TBD)
        radius: radius of the pin (unit of measure TBD)
        J_to_eV: conversion from Joule to electronVolts
        tally_dict: dictionary of openmc tallies

    functions:
        eval: returns the power density, z coordinates, power density std. dev., Area (?)
        normalisation_z: normalise heat 
        getSpectrum: returns the enrgy spectrum (mean and uncertainty) and energy list

    '''

    def __init__(self, n_div: int, power: float, mesh_size: int, length: float, radius: float,
                      J_to_eV: float, tally_dict: dict):
        self.n_div = n_div
        self.power = power
        self.mesh_size = mesh_size
        
        #self.pin_length = pin_length
        self.length = length
        self.radius = radius
        #self.pin_radius = pin_radius
        
        self.J_to_eV = J_to_eV
        self.tally_dict = tally_dict
        
    def eval_from_fission(self, sp, i, Ef = 200e6):
 
        # INTEGRAL TALLY
        tally_integral = sp.get_tally(name = self.tally_dict['integral'])

        # Integral fission energy
        heating_integral = tally_integral.get_slice(scores=['kappa-fission']) 
        Qp = float(heating_integral.mean) # measure of the fission energy (eV/src) 

        ### Axial scores
        tally_mesh = sp.get_tally(name = self.tally_dict['mesh_z'])

        # Fission reaction rate (z)
        fissionZ = tally_mesh.get_slice(scores=['fission']) # fission comes in (fissions/source)

        # Power density (z)
        dz = self.length / self.mesh_size
        z = np.arange(-self.length/2, self.length/2., dz )
        
        
        RR_Z, uRR_Z, Area = self.normalisation_z(fissionZ, Qp) # fissions normalized to (fissions / cm3 s)

    
        q3 = RR_Z*Ef*self.J_to_eV # (fiss/cm3 s * eV/fiss * J/eV) = (W/cm3)
        q3std = uRR_Z*Ef*self.J_to_eV

        return q3, z, q3std, Area
    
    def eval_from_heating_local(self, sp, i):
 
        # INTEGRAL TALLY
        tally_integral = sp.get_tally(name = self.tally_dict['integral'])

        # Integral fission energy
        heating_integral = tally_integral.get_slice(scores=['heating-local']) 
        Qp = float(heating_integral.mean) # measure of the fission energy (eV/src) 

        ### Axial scores
        tally_mesh = sp.get_tally(name = self.tally_dict['mesh_z'])

        # Fission reaction rate (z)
        heating_local = tally_mesh.get_slice(scores=['heating-local']) # heating comes in (eV/source)
        RR_Z, uRR_Z, Area = self.normalisation_z(heating_local, Qp) # heating normalized to (eV / cm3 s)

        # Power density (z)
        dz = self.length / self.mesh_size
        z = np.arange(-self.length/2, self.length/2., dz )
    
        q3 = RR_Z*self.J_to_eV # (fiss/cm3 s * eV/fiss * J/eV) = (W/cm3)
        q3std = uRR_Z*self.J_to_eV

        return q3, z, q3std, Area
    
    

    

    # very important function :)
    def normalisation_z(self, qty_to_norm, Qp):
        
        H1 = self.J_to_eV * Qp # (J/source)
        Vol = (self.radius**2) * self.length * np.pi # pin volume (cm3)
        
        f = self.mesh_size * self.power / (H1 * Vol) # Normalization factor ( source/(s cm3) )
        #f = self.power / (H1 * Vol) # Normalization factor ( source/(s cm3) )
        
        # Put quantities in the right shape
        q_mean = qty_to_norm.mean # (fissions / source)
        q_mean.shape = self.mesh_size 
        
        q_std = qty_to_norm.std_dev
        q_std.shape = self.mesh_size 

        q_mean_normalised = q_mean * f # (fissions / (s cm3))
        q_std_normalised = q_std * f 
        
        return q_mean_normalised, q_std_normalised, Vol/self.length

    def getSpectrum(self, sp, phiE_list):
        
        ### Energy score
        tally_energy = sp.get_tally(name = self.tally_dict['spectrum'])

        # Flux in energy
        energy_filter = tally_energy.filters[0]
        energies = energy_filter.bins[:, 0]

        # Get the flux values
        mean = tally_energy.get_values(value='mean').ravel()
        uncertainty = tally_energy.get_values(value='std_dev').ravel()

        EnergyStairs = np.zeros(len(energies)+1)
        EnergyStairs[0:-1] = energies
        EnergyStairs[-1] = energies[-1]

        phiE_list[0].append(np.array(mean))
        phiE_list[1].append(np.array(uncertainty))

        return phiE_list, EnergyStairs



def UpdateParticles(new_folder, sn):
    '''
    UpdateParticles update the number of neutrons for openmc simulation in the setting.xml file
    requires the direcotry of new folder, and the new number of particles
    '''

    file = open (new_folder+'/settings.xml', "r")
    list_lines = file.readlines()
    list_lines[3] = ( "  <particles>"+str(int(sn))+"</particles>\n" )
    file = open (new_folder+'/settings.xml', "w")
    file.writelines(list_lines)
    file.close()
    
    
def RemoveFolders(path=None):
    '''
    Remove folders given the path
    '''
    if path is None:
        path = os.getcwd()
    folder_to_check = path + '/build_xml'
    n_folders = len(next(os.walk(folder_to_check))[1])
    
    if n_folders>1:
        for k in range(n_folders-1):
            folder_to_delete = folder_to_check + '/it' + str(k+1)
            os.system('rm -r ' + folder_to_delete)

def saveScalarFieldToFile(domain,scalar_field,file_name,field_name):
    # Temperature storage in .h5 format
    with XDMFFile(domain.comm, file_name+'.xdmf', "w") as loc:
        scalar_field.name = field_name
        loc.write_mesh(domain)
        loc.write_function(scalar_field)



def dataOverLine(domain: dolfinx.mesh.Mesh, points: np.ndarray):
    """
    This function can be used to extract data along a line defined by the variables `points`, crossing the domain.
 
    Parameters
    ----------
    domain  : dolfinx.mesh.Mesh
        Domain to extract data from.
    points : np.ndarray 
        Points listing the line from which data are extracted.

    Returns
    -------
    xPlot : np.ndarray 
        Coordinate denoting the cell from which data are extracted.
    cells : list
        List of cells of the mesh.
    """
    bb_tree = geometry.BoundingBoxTree(domain, domain.topology.dim)
    cells = []
    points_on_proc = []
    cell_candidates = geometry.compute_collisions(bb_tree, points.T)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, points.T)
    for i, point in enumerate(points.T):
        if len(colliding_cells.links(i))>0:
            points_on_proc.append(point)
            cells.append(colliding_cells.links(i)[0])
    xPlot = np.array(points_on_proc, dtype=np.float64)

    return xPlot, cells