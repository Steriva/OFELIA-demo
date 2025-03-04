import numpy as np
from scipy.interpolate import interp1d

import dolfinx
from dolfinx import fem
from dolfinx.fem import (Function, FunctionSpace, assemble_scalar, form, dirichletbc, locate_dofs_topological, locate_dofs_geometrical, assemble_scalar)
import ufl
from ufl import SpatialCoordinate, inner, grad
from petsc4py import PETSc
from mpi4py import MPI
from pyXSteam.XSteam import XSteam

import sys


class robin_class():
    def __init__(self,h,Tb,marker : int,name : str = None):
        self.h = h
        self.Tb = Tb #lambda function
        self.marker = marker 
        self.name = name
        self.type = "robin"

class dirichlet_class():
    def __init__(self,TD : float,marker : int,name : str = None):
        self.TD = TD  #int or float
        self.marker = marker
        self.name = name
        self.type = "dirichlet"

class neumann_class():
    def __init__(self,q,marker : int,name : str = None):
        self.q = q  #int or float
        self.marker = marker
        self.name = name
        self.type = "neumann"



class thermal_solver():
    '''
    thermal_solver is a class

    it is used to solve the thermal problem with FEniCSx

    attributes:
        domain: FEM domain
        ct: cells tag for the subdomains (dolfinx.cpp.mesh.MeshTags_int32)
        ft: faces tag for the bounaries (dolfinx.cpp.mesh.MeshTags_int32)
        physical_param: dictionary of physical parameters (th_cond (type?), htc (type?))
        regions_markers: list of markers for physical regions
        robin_mark: Surfaces' marks where robin bc is applied
        degree: Order of the lagrange elements, default 1 (integer)
        type_of_simmetry: string or None, define the type of symmetry of the thermal system (ex: "cyl" for cylindrical)

    functions:
        assemble: assemble the matrix (direct = False by default)
        solve: solve the system. Requires power distribution and boundary temperature (both interpolant)
        computeSolidAverageT: Compute average temperature in a region (defined by marker is_region) from temparture field T_sol. Region is subdivided in slices defined by "slices"  (np.array)
        extract_2D_data: Returns a dictionary with 2d data (mesh and temperature) along radial and axial directions
    '''


    def __init__(self, domain: dolfinx.mesh.Mesh, ct: dolfinx.cpp.mesh.MeshTags_int32, ft: dolfinx.cpp.mesh.MeshTags_int32,
                 physical_param : dict, regions_markers: list, degree : int = 1,type_of_symmetry = None):
    
        # Storing domain, cell tags, face tags and the functional space
        self.domain = domain
        self.ct = ct
        self.ft = ft
        self.funSpace = FunctionSpace(domain, ("Lagrange", degree))
        self.Qn = FunctionSpace(domain, ("DG", 0))

        self.gdim = self.domain.geometry.dim
        self.fdim = self.gdim - 1

        self.type_of_symmetry = type_of_symmetry # --- select the type of symmetry --- rb
        
        # Defining physical parameters and labels for the regions and boundaries        
        self.regions = regions_markers # the first element (in position [0] must be the fuel)
        self.phys_param = physical_param

        
        
        
        # --- Thermal conductivity and heat source --- rb
        self.k = Function(self.Qn)  # thermal conductivity
        self.q  = Function(self.Qn) # heat source
        for idx, regionI in enumerate(self.regions):
            cells = self.ct.find(regionI)
            self.k.x.array[cells] = self.phys_param['th_cond'][idx]                                 #assign thermal conductivity


        
        

        # Definition of the trial and test space
        self.T = ufl.TrialFunction(self.funSpace)
        self.v = ufl.TestFunction(self.funSpace)
        self.solution = Function(self.funSpace)
        
        
        
        
        # Definition of the surface and volume element for cylindrical coordinate system
        self.ds = ufl.Measure('ds', domain=self.domain, subdomain_data=self.ft)
        self.dx = ufl.Measure('dx', domain=self.domain, subdomain_data=self.ct)
        

        # --- get spatial coordinate depending on geo dimension --- rb
        if self.gdim == 2: 
            [self.z_, self.r_] = SpatialCoordinate(domain)
        elif self.gdim == 3:
            [self.z_, self.x_, self.y_] = SpatialCoordinate(domain) # --- x,y,z (ma non serve)
        else:
            print("Geo dimension not valid, exiting...")
            sys.exit(0)


  

    def applyBCs(self, bcs):
        self.all_bcs = bcs
        self.robin_markers = []
        self.dirichlet_markers = []
        self.neumann_markers = []

        T_b_fun = []
        TD = []
        q_neumann = []
        for bc in self.all_bcs:
            if bc.type == "robin":
                self.robin_markers.append(bc.marker)
                T_b_fun.append(bc.Tb)
            elif bc.type == "dirichlet":
                self.dirichlet_markers.append(bc.marker)
                TD.append(bc.TD)
            elif bc.type == "neumann":
                self.neumann_markers.append(bc.marker)
                q_neumann.append(bc.q)
            else:
                print("unkwonw type of bc")


        # --- initialise dirichlet bcs --- rb
        self.dirichlet_bcs = []
        # --- add dirichlet boundary condition --- rb

        # configure dirichlet boundary conditions
        if self.dirichlet_markers is not None:
            self.TD = [Function(self.funSpace)]*len(self.dirichlet_markers)
            for ii in range(len(self.dirichlet_markers)):
                temp_td = self.TD[ii].copy() #required to avoid to overwrite bc
                temp_td.x.set(TD[ii])
                self.TD[ii] = temp_td.copy()
            self.dirichlet_bcs = [dirichletbc(self.TD[ii], locate_dofs_topological(self.funSpace, self.fdim, self.ft.find(self.dirichlet_markers[ii]))) for ii in range(len(self.dirichlet_markers))]
            
        # --- h deve essere 0 se non ci sono robin markers
        #self.h = fem.Constant(self.domain, PETSc.ScalarType(0))
        if self.robin_markers is not None:
            self.h = [fem.Constant(self.domain, PETSc.ScalarType(self.all_bcs[ii].h)) for ii in range(len(self.robin_markers))] # heat transfer coefficient

        # Definition of the bulk temperature of the coolant 
        self.T_b = [Function(self.funSpace)]*len(self.robin_markers)
        # --- assign bulk temperature bc (robin) --- rb
        if len(T_b_fun) > 0:
            for ii in range(len(self.T_b)):
                temp_tb = self.T_b[ii].copy()
                temp_tb.interpolate(lambda x: T_b_fun[ii](x))#T_b_fun[ii](x[0], x[1], x[2])) 
                self.T_b[ii] = temp_tb.copy()
            #[self.T_b[ii].interpolate(lambda x: T_b_fun[ii](x[0], x[1], x[2])) for ii in range(len(self.T_b))]

        if self.neumann_markers is not None:
            self.q_neumann = [fem.Constant(self.domain, PETSc.ScalarType(self.all_bcs[ii].q)) for ii in range(len(self.neumann_markers))] # heat transfer coefficient




    def assemble(self, direct : bool = False):

        #create LHS and RHS for 3D general case
        self.left_side  = (inner(self.k * grad(self.T), grad(self.v)) * self.dx) 
        self.right_side = (inner(self.q, self.v) * self.dx())
        for idx in range(len(self.robin_markers)): 
            self.left_side += inner(self.h[idx] * self.T, self.v) * self.ds(self.robin_markers[idx]) 
            self.right_side += inner(self.h[idx] * self.T_b[idx], self.v) * self.ds(self.robin_markers[idx])
        for idx in range(len(self.neumann_markers)):
            self.right_side += inner(self.q_neumann[idx], self.v) * self.ds(self.neumann_markers[idx]) 
            

        #check simmetry
        if self.gdim == 2:
            if self.type_of_symmetry == "cyl": # --- cylindrical symmetry --- rb
                self.left_side  = (inner(self.k * grad(self.T), grad(self.v)) * np.abs(self.r_) * self.dx) 
                self.right_side = (inner(self.q, self.v) * np.abs(self.r_) * self.dx())
                for idx in range(len(self.robin_markers)): 
                    self.left_side += inner(self.h[idx] * self.T, self.v) * self.ds(self.robin_markers[idx]) 
                    self.right_side += inner(self.h[idx] * self.T_b[idx], self.v) * self.ds(self.robin_markers[idx]) 
                for idx in range(len(self.neumann_markers)):
                    self.right_side += inner(self.q_neumann[idx], self.v) * self.ds(self.neumann_markers[idx]) 
       
        # --- OLD --- rb
        #############################
        #self.a = form(self.left_side)
        #self.L = form(self.right_side)
        
        # Creating and storing the matrices
        #self.A = fem.petsc.assemble_matrix(self.a)
        #self.A.assemble()
        #self.b = fem.petsc.create_vector(self.L)
        ############################# 
        # --- new --- rb 
        self.a  = form(self.left_side)
        self.L  = form(self.right_side)
        
        self.A = fem.petsc.create_matrix(self.a)
        self.A.zeroEntries()
        
        # Add Dirichlet
        if len(self.dirichlet_bcs) > 0:
            fem.petsc.assemble_matrix(self.A, self.a, self.dirichlet_bcs)
        else:
            fem.petsc.assemble_matrix(self.A, self.a)
    
        self.A.assemble()  
        self.b = fem.petsc.create_vector(self.L)
        #############################
        
        # Definition of the solver
        self.solver = PETSc.KSP().create(self.domain.comm)
        self.solver.setOperators(self.A)
        
        if direct:
            self.solver.setType(PETSc.KSP.Type.PREONLY)
            self.solver.getPC().setType(PETSc.PC.Type.LU)
        else:
            self.solver.setType(PETSc.KSP.Type.CG)
            self.solver.getPC().setType(PETSc.PC.Type.SOR)
        

    def solve(self, power_fun : list, power_integral): # power_fun should be a callable interpolant

        
        Qheat = Function(self.Qn)
        
        
        # set power density
        for idx, regionI in enumerate(self.regions): 
            cells = self.ct.find(regionI)
            points_coordinates = self.Qn.tabulate_dof_coordinates()                                   #get points coordinates
            Qheat.x.array[cells] = power_fun[idx](points_coordinates[cells].T)                       #assign heat source
        
        #TO CHECK
        if self.type_of_symmetry == "cyl":
            Q_norm_factor = power_integral/assemble_scalar(form(Qheat * np.abs(self.r_) * self.dx))
        else:
            Q_norm_factor = power_integral/assemble_scalar(form(Qheat * self.dx))
        
        self.q.x.array[:] = Q_norm_factor*Qheat.x.array

        #print(assemble_scalar(form(self.q*self.dx)))

        # Updating the rhs vector
        with self.b.localForm() as loc:
            loc.set(0)
        fem.petsc.assemble_vector(self.b, self.L)
        
        

        # add dirichled bc if present
        # --- new --- rb
        if len(self.dirichlet_bcs) > 0:
            # Apply Dirichlet boundary condition to the vector
            fem.petsc.apply_lifting(self.b, [self.a], [self.dirichlet_bcs])
            self.b.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
            fem.petsc.set_bc(self.b, self.dirichlet_bcs)
        # --- end new --- rb
            

        # Solving the linear system
        self.solver.solve(self.b, self.solution.vector)
        self.solution.x.scatter_forward()
        
        return self.solution.copy(), self.q.copy()
    

    def computeSolidAverageT_X(self, is_region: int, T_sol: dolfinx.fem.Function, slices: np.ndarray):

        # Definition of the function and the form for the integration
        sliceID = Function(self.Qn)
        integral_form = sliceID * T_sol * self.dx(is_region)
        domain_form   = sliceID * self.dx(is_region)
        
        # Computing the average temperature
        aveT = np.zeros([len(slices)-1])
        for ii in range(len(slices)-1):
            bounds = np.array([slices[ii], slices[ii+1]])
            
            
            sliceID.interpolate(lambda x: np.piecewise(x[0], [x[0]<bounds[0],
                                                            np.logical_and(x[0]>=bounds[0], x[0]<=bounds[1]),
                                                            x[0]>=bounds[1]],
                                                            [0., 1., 0.]))
            

            temperatureIntegral = self.domain.comm.allreduce(assemble_scalar(form(integral_form)), op=MPI.SUM) 
            volumeIntegral = self.domain.comm.allreduce(assemble_scalar(form(domain_form)), op=MPI.SUM)
            aveT[ii] = temperatureIntegral/volumeIntegral
            
        return aveT
    

    def computeSolidAverageT_Y(self, is_region: int, T_sol: dolfinx.fem.Function, slices: np.ndarray):

        # Definition of the function and the form for the integration
        sliceID = Function(self.Qn)
        integral_form = sliceID * T_sol * self.dx(is_region)
        domain_form   = sliceID * self.dx(is_region)
        
        # Computing the average temperature
        aveT = np.zeros([len(slices)-1])
        for ii in range(len(slices)-1):
            bounds = np.array([slices[ii], slices[ii+1]])
            
            
            sliceID.interpolate(lambda x: np.piecewise(x[1], [x[1]<bounds[0],
                                                            np.logical_and(x[1]>=bounds[0], x[1]<=bounds[1]),
                                                            x[1]>=bounds[1]],
                                                            [0., 1., 0.]))
            

            temperatureIntegral = self.domain.comm.allreduce(assemble_scalar(form(integral_form)), op=MPI.SUM) 
            volumeIntegral = self.domain.comm.allreduce(assemble_scalar(form(domain_form)), op=MPI.SUM)
            aveT[ii] = temperatureIntegral/volumeIntegral
            
        return aveT


    def computeSolidAverageT_Z(self, is_region: int, T_sol: dolfinx.fem.Function, slices: np.ndarray):

        # Definition of the function and the form for the integration
        sliceID = Function(self.Qn)
        integral_form = sliceID * T_sol * self.dx(is_region)
        domain_form   = sliceID * self.dx(is_region)
        
        # Computing the average temperature
        aveT = np.zeros([len(slices)-1])
        for ii in range(len(slices)-1):
            bounds = np.array([slices[ii], slices[ii+1]])
            
            
            sliceID.interpolate(lambda x: np.piecewise(x[2], [x[2]<bounds[0],
                                                            np.logical_and(x[2]>=bounds[0], x[2]<=bounds[1]),
                                                            x[2]>bounds[1]],
                                                            [0., 1., 0.]))
            

            temperatureIntegral = self.domain.comm.allreduce(assemble_scalar(form(integral_form))) 
            volumeIntegral = self.domain.comm.allreduce(assemble_scalar(form(domain_form)), op=MPI.SUM)
            aveT[ii] = temperatureIntegral/volumeIntegral

            
            
        return aveT

    
    
    def extract_2D_data(self, T, L, R, Nx = 400, Ny = 100):
        
        x_grid = np.linspace(-L/2, L/2, Nx)
        y_grid = np.linspace(-R, R, Ny)

        T_matrix = np.zeros((Nx, Ny))

        for ii in range(Nx):
            points = np.zeros((3, Ny))
            points[0, :] = x_grid[ii]
            points[1, :] = y_grid

            bb_tree = dolfinx.geometry.BoundingBoxTree(self.domain, self.domain.topology.dim)
            cells = []
            points_on_proc = []
            cell_candidates = dolfinx.geometry.compute_collisions(bb_tree, points.T)
            colliding_cells = dolfinx.geometry.compute_colliding_cells(self.domain, cell_candidates, points.T)
            for i, point in enumerate(points.T):
                if len(colliding_cells.links(i))>0:
                    points_on_proc.append(point)
                    cells.append(colliding_cells.links(i)[0])
            xPlot = np.array(points_on_proc, dtype=np.float64)

            T_matrix[ii, :] = T.eval(xPlot, cells).flatten()
        
        X, Y = np.meshgrid(x_grid, y_grid)
        res2d = dict()
        res2d['xgrid'] = x_grid
        res2d['ygrid'] = y_grid
        res2d['X'] = X
        res2d['Y'] = Y
        res2d['T'] = T_matrix
        return res2d






class thermal_inputs():
    '''
    thermal_inputs is a class

    Get inputs and computes thermophysical paramters

    attributes: 
        coolant_T: coolant temperature, float (input)
        coolant_p: coolant pressure, float (input)

        rho: density computed from steam tables
        cp: constant pressure specific heat computed from steam tables
        k: thermal conductivity computed from steam tables
        mu: viscosity computed from steam tables
        Pr: computed Prantdl number

    functions: 
        compute_htc: compute heat transfer coefficient from Dittus-Boelter correlation. requires pin pitch (pitch), cladding outer radius (clad_or), inlet velocity (u_in)
        mapping_q3_Tb: map the power density (from openmc) and computes fluid temperature. returns q and temperature lambda fanctions and results of mapping (dict)
        computeWaterAverageT: Computes average liquid temperature given some slices
    '''

    def __init__(self, coolant_T: float, coolant_p: float):
        
        # Storing coolant properties
        self.coolant_T = coolant_T
        self.coolant_p = coolant_p
        

        # Computing flow thermophysical properties
        steamTable = XSteam(XSteam.UNIT_SYSTEM_MKS)
        self.rho = steamTable.rho_pt(self.coolant_p, self.coolant_T - 273.15) / 1e3 # g/cm3 
        self.cp  = steamTable.Cp_pt( self.coolant_p, self.coolant_T - 273.15) # J/g K
        self.k   = steamTable.tc_pt( self.coolant_p, self.coolant_T - 273.15) / 100 # W/cm-K
        self.mu  = steamTable.my_pt( self.coolant_p, self.coolant_T - 273.15) * 10  # g / cm s^2
        self.Pr  = self.cp * self.mu / self.k

    

    def compute_htc(self, pitch: float, clad_or: float, u_in: float):
        
        
        # Computing hydraulic diameter
        flow_area = pitch**2 - np.pi * clad_or**2
        Dh = 4 * flow_area / (2 * np.pi * clad_or)

        # Computing equivalent Reynolds Number
        Re = self.rho * u_in * Dh / self.mu
        
        # The convective HTC is computed using the Dittus-Boelter correlation
        h = self.k / Dh * (0.023 * Re**0.8 * self.Pr**0.4) # W/cm2 - K
        self.u_in = u_in
        self.flow_area = flow_area
        
        return h
    
    def mapping_q3(self,  z_omc: np.ndarray, q3_omc: np.ndarray,L: float, L_active: float ):
        # Mapping q3 from OpenMC to scipy.interpolate
        q3_fuel = interp1d(z_omc, q3_omc, kind='linear', fill_value="extrapolate")
        dimz_full = z_omc.shape[0]
        
        if z_omc.shape[0] < 100:  #if too coarse refine 
            dimz_full = 100
        else:
            dimz_full = z_omc.shape[0]+100
        
        z_full = np.linspace(-L/2, L/2, dimz_full)
        q3_full = np.zeros_like(z_full)
        
        for kk in range(len(z_full)):
            if z_full[kk] < -L_active/2:
                value1 = 0.
            elif (( z_full[kk] >= -L_active/2 ) & ( z_full[kk] <= L_active/2 )):
                value1 = q3_fuel(z_full[kk])
            else:
                value1 = 0.
            q3_full[kk] = value1
               
        q3_fuel = interp1d(z_full, q3_full, kind='linear', fill_value="extrapolate")

        return lambda x: q3_fuel(x[2]) + 0.0 * x[1] +0.0*x[0]
    

    def mapping_q3_Tb_1D(self, z_omc: np.ndarray, q3_omc: np.ndarray, Tin: float, L: float, L_active: float, fuel_or: float):
        
        # Mapping q3 from OpenMC to scipy.interpolate
        q3_fuel = interp1d(z_omc, q3_omc, kind='linear', fill_value="extrapolate")
        dimz_full = z_omc.shape[0]
        if z_omc.shape[0] < 100:
            dimz_full = 100
        else:
            dimz_full = z_omc.shape[0]+100
        
        z_full = np.linspace(-L/2, L/2, dimz_full)
        q3_full = np.zeros_like(z_full)
        
        for kk in range(len(z_full)):
            if z_full[kk] < -L_active/2:
                value1 = 0.
            elif (( z_full[kk] >= -L_active/2 ) & ( z_full[kk] <= L_active/2 )):
                value1 = q3_fuel(z_full[kk])
            else:
                value1 = 0.
            q3_full[kk] = value1
               
        q3_fuel = interp1d(z_full, q3_full, kind='linear', fill_value="extrapolate")
        
        # Energy balance to compute the bulk temperature of the fuel
        m_cp = (self.rho * self.flow_area * self.u_in) * self.cp # g/s * J/g/K # cp at 300 C and 155 bar

        T_b = np.zeros((len(z_full), ))
        T_b[0] = Tin
        for ii in range(1, len(z_full)):
            T_b[ii] = T_b[0] + 1. / m_cp * np.pi * fuel_or**2 * np.trapz(q3_fuel(z_full[:ii+1]), z_full[:ii+1])
        
        # Creating Tb interpolant
        self.T_b_fun = interp1d(z_full, T_b, kind='linear',fill_value='extrapolate')
        
        # Storing the results
        mapping_res = dict()
        mapping_res['z'] = z_full
        mapping_res['q3'] = q3_fuel
        mapping_res['T_bulk'] = self.T_b_fun

        #return lambda x,y: q3_fuel(x) + 0.0 * y, lambda x,y: self.T_b_fun(x) + 0.0 * y, mapping_res
        return lambda x: q3_fuel(x[2]) + x[0]*0. + x[1]*0, lambda x: self.T_b_fun(x[2]) + x[0]*0. + x[1]*0, mapping_res

    def mapping_q3_Tb_OLD(self, z_omc: np.ndarray, q3_omc: np.ndarray, Tin: float, L: float, L_active: float, fuel_or: float):
        
        # Mapping q3 from OpenMC to scipy.interpolate
        q3_fuel = interp1d(z_omc, q3_omc, kind='linear', fill_value="extrapolate")
        dimz_full = z_omc.shape[0]
        if z_omc.shape[0] < 100:
            dimz_full = 100
        else:
            dimz_full = z_omc.shape[0]+100
        
        z_full = np.linspace(-L/2, L/2, dimz_full)
        q3_full = np.zeros_like(z_full)
        
        for kk in range(len(z_full)):
            if z_full[kk] < -L_active/2:
                value1 = 0.
            elif (( z_full[kk] >= -L_active/2 ) & ( z_full[kk] <= L_active/2 )):
                value1 = q3_fuel(z_full[kk])
            else:
                value1 = 0.
            q3_full[kk] = value1
               
        q3_fuel = interp1d(z_full, q3_full, kind='linear', fill_value="extrapolate")
        
        # Energy balance to compute the bulk temperature of the fuel
        m_cp = (self.rho * self.flow_area * self.u_in) * self.cp # g/s * J/g/K # cp at 300 C and 155 bar

        T_b = np.zeros((len(z_full), ))
        T_b[0] = Tin
        for ii in range(1, len(z_full)):
            T_b[ii] = T_b[0] + 1. / m_cp * np.pi * fuel_or**2 * np.trapz(q3_fuel(z_full[:ii+1]), z_full[:ii+1])
        
        # Creating Tb interpolant
        self.T_b_fun = interp1d(z_full, T_b, kind='linear',fill_value='extrapolate')
        
        # Storing the results
        mapping_res = dict()
        mapping_res['z'] = z_full
        mapping_res['q3'] = q3_fuel
        mapping_res['T_bulk'] = self.T_b_fun

        return lambda x,y: q3_fuel(x) + 0.0 * y, lambda x,y: self.T_b_fun(x) + 0.0 * y, mapping_res
    
    def computeWaterAverageT(self, slices: np.ndarray, dim_z: int):
        
        # Definition of the average and the z for the integration
        aveT = np.zeros([len(slices)-1])
        z = np.linspace(min(slices), max(slices), dim_z)

        # Computing average temperature
        for ii in range(len(slices)-1):
            bounds = np.array([slices[ii], slices[ii+1]])
            sliceID = lambda x: np.piecewise(x, [x<bounds[0],
                                            np.logical_and(x>=bounds[0], x<=bounds[1]),
                                            x>=bounds[1]],
                                            [0., 1., 0.])
                            
            aveT[ii] = np.trapz( self.T_b_fun(z) * sliceID(z), z) / np.trapz( sliceID(z), z) 
        return aveT
    
