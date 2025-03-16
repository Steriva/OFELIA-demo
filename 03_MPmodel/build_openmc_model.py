from math import log10, pi

import matplotlib.pyplot as plt
import numpy as np
import openmc

class MyOpenMCmodel():
    
    def __init__(self, path_to_run):
        self.path = path_to_run
        self.mat_dict = None
        self.cells_list = None
        self.settings = None
        self.tallies = None
        self.plots = None

    def clean_model(self):
        """Reset model components to None."""
        self.mat_dict = None
        self.cells_list = None
        self.settings = None
        self.tallies = None
        self.plots = None

    def create_materials(self, n_div, Tin, Tout): # create Dictionary which contains the used materials

        mat_dict = dict()
        mat_dict['coolant'] = [] # list of water layers
        mat_dict['fuel'] = [] # list of fuel slices
        mat_dict['non_updated'] = [] # list of materials whose temperature is constant (i.e., He, Clad)

        # water0 and fuel0 are used for cloning the real materials that will be put in the system
        idx_mat = 0
        mat_dict['non_updated'].append(openmc.Material(name='water0'))
        mat_dict['non_updated'][idx_mat].set_density('g/cm3', 0.75)
        mat_dict['non_updated'][idx_mat].add_element('H', 2)
        mat_dict['non_updated'][idx_mat].add_element('O',1)
        mat_dict['non_updated'][idx_mat].add_s_alpha_beta('c_H_in_H2O')

        # Cloning water materials
        for kk in range(n_div):
            mat_dict['coolant'].append(mat_dict['non_updated'][idx_mat].clone())
            mat_dict['coolant'][kk].name = 'coolant_'+str(kk+1)    

        idx_mat += 1
        mat_dict['non_updated'].append(openmc.Material(name='fuel0'))
        mat_dict['non_updated'][idx_mat].set_density('g/cm3', 10.45)
        mat_dict['non_updated'][idx_mat].add_nuclide('U235', 9.3472e-4, 'ao')
        mat_dict['non_updated'][idx_mat].add_nuclide('U238', 2.1523e-2, 'ao')
        mat_dict['non_updated'][idx_mat].add_nuclide('U234', 9.1361e-6, 'ao')
        mat_dict['non_updated'][idx_mat].add_nuclide('O16', 4.4935e-02, 'ao')
        mat_dict['non_updated'][idx_mat].temperature = 800

        # Cloning fuel materials
        for kk in range(n_div):
            mat_dict['fuel'].append(mat_dict['non_updated'][idx_mat].clone()) 
            mat_dict['fuel'][kk].name = 'fuel_'+str(kk+1)
            
        idx_mat += 1
        mat_dict['non_updated'].append(openmc.Material(name='Helium for gap'))
        mat_dict['non_updated'][idx_mat].set_density('g/cm3', 0.001598)
        mat_dict['non_updated'][idx_mat].add_element('He', 2.4044e-4)
        mat_dict['non_updated'][idx_mat].temperature = 600

        idx_mat += 1
        mat_dict['non_updated'].append(openmc.Material(name="Zirc4"))
        mat_dict['non_updated'][idx_mat].set_density('g/cm3', 6.44)
        mat_dict['non_updated'][idx_mat].add_nuclide('O16', 1.192551825E-03, 'wo')
        mat_dict['non_updated'][idx_mat].add_nuclide('O17', 4.82878E-07, 'wo')
        mat_dict['non_updated'][idx_mat].add_nuclide('Cr50', 4.16117E-05, 'wo')
        mat_dict['non_updated'][idx_mat].add_nuclide('Cr52', 8.34483E-04, 'wo')
        mat_dict['non_updated'][idx_mat].add_nuclide('Cr53', 9.64457E-05, 'wo')
        mat_dict['non_updated'][idx_mat].add_nuclide('Cr54', 2.446E-05, 'wo')
        mat_dict['non_updated'][idx_mat].add_nuclide('Fe54', 1.1257E-04, 'wo')
        mat_dict['non_updated'][idx_mat].add_nuclide('Fe56', 1.8325E-03, 'wo')
        mat_dict['non_updated'][idx_mat].add_nuclide('Fe57', 4.3077E-05, 'wo')
        mat_dict['non_updated'][idx_mat].add_nuclide('Fe58', 5.833E-06, 'wo')
        mat_dict['non_updated'][idx_mat].add_nuclide('Zr90', 4.9786E-01, 'wo')
        mat_dict['non_updated'][idx_mat].add_nuclide('Zr91', 1.0978E-01, 'wo')
        mat_dict['non_updated'][idx_mat].add_nuclide('Zr92', 1.6964E-01, 'wo')
        mat_dict['non_updated'][idx_mat].add_nuclide('Zr94', 1.7566E-01, 'wo')
        mat_dict['non_updated'][idx_mat].add_nuclide('Zr96', 4.28903E-02, 'wo')
        mat_dict['non_updated'][idx_mat].add_nuclide('Sn116', 1.98105E-03, 'wo')
        mat_dict['non_updated'][idx_mat].add_nuclide('Sn117', 1.05543E-03, 'wo')
        mat_dict['non_updated'][idx_mat].add_nuclide('Sn119', 1.20069E-03, 'wo')
        mat_dict['non_updated'][idx_mat].add_nuclide('Sn120', 4.5922E-03, 'wo')
        mat_dict['non_updated'][idx_mat].add_nuclide('Sn122', 6.63497E-04, 'wo')
        mat_dict['non_updated'][idx_mat].add_nuclide('Sn124', 8.43355E-04, 'wo')
        mat_dict['non_updated'][idx_mat].temperature = 600 # [K]

        idx_mat += 1
        mat_dict['non_updated'].append(openmc.Material(name="Water inlet"))
        mat_dict['non_updated'][idx_mat].set_density('g/cm3', 0.996557)
        mat_dict['non_updated'][idx_mat].add_element('H', 2)
        mat_dict['non_updated'][idx_mat].add_element('O', 1)
        mat_dict['non_updated'][idx_mat].add_s_alpha_beta('c_H_in_H2O')
        mat_dict['non_updated'][idx_mat].temperature = Tin

        idx_mat += 1
        mat_dict['non_updated'].append(openmc.Material(name="Water outlet"))
        mat_dict['non_updated'][idx_mat].set_density('g/cm3', 0.996557)
        mat_dict['non_updated'][idx_mat].add_element('H', 2)
        mat_dict['non_updated'][idx_mat].add_element('O', 1)
        mat_dict['non_updated'][idx_mat].add_s_alpha_beta('c_H_in_H2O')
        mat_dict['non_updated'][idx_mat].temperature = Tout

        self.mat_dict = mat_dict

    def create_geometry(self, fuel_or, clad_ir, clad_or, pitch, l_active, n_div, end_domain_top, end_domain_bot, plug_length):
        # Create cylindrical surfaces
        Fuel_or = openmc.ZCylinder(r=fuel_or, name='Fuel OR')
        Clad_ir = openmc.ZCylinder(r=clad_ir, name='Clad IR')
        Clad_or = openmc.ZCylinder(r=clad_or, name='Clad OR')

        # Create planes for the channel
        square_side = pitch
        east_boundary = openmc.XPlane(x0= square_side/2, boundary_type = 'reflective', name='right boundary')
        west_boundary = openmc.XPlane(x0=-square_side/2, boundary_type = 'reflective', name='left  boundary')
        north_boundary = openmc.YPlane(y0=square_side/2, boundary_type = 'reflective', name='north boundary')
        south_boundary = openmc.YPlane(y0=-square_side/2, boundary_type = 'reflective', name='south boundary')

        z0_bottom = -l_active/2
        z0_top = l_active/2
        dz = (z0_top-z0_bottom)/n_div

        # Create boundaries for the 3D pin
        top_domain = openmc.ZPlane(z0=end_domain_top, boundary_type = 'vacuum', name ='top domain')
        top_clad = openmc.ZPlane(z0=z0_top+plug_length,  name ='top clad')
        top_active= openmc.ZPlane(z0=z0_top, name='top boundary')

        bot_active= openmc.ZPlane(z0=z0_bottom,name='bot boundary')
        bot_clad = openmc.ZPlane(z0=z0_bottom-plug_length, name ='bot clad')
        bot_domain = openmc.ZPlane(z0=end_domain_bot, boundary_type = 'vacuum', name ='bot domain')


        # Create planes for material subdivision
        z_planes = []

        for j in range(n_div-1):#7 piani in mezzo
            plane = openmc.ZPlane(z0=z0_bottom + dz*(j+1))
            z_planes.append(plane)


        # Create cells and assigning materials to regions in ascendent order
        cells_list = []
        for div in range(n_div):
            if div==0: #bottom zone
                fuel_cell=openmc.Cell(fill=self.mat_dict['fuel'][div], region = -Fuel_or & +bot_active & -z_planes[div], name='fuelzone' + str(div))
                water_cell=openmc.Cell(fill=self.mat_dict['coolant'][div], region=+Clad_or & +west_boundary & -east_boundary  & +south_boundary & -north_boundary  & +bot_active & -z_planes[div], name='waterzone' + str(div))
                
            elif div==(n_div-1):
                fuel_cell=openmc.Cell(fill=self.mat_dict['fuel'][div], region = -Fuel_or & +z_planes[div-1] & -top_active, name='fuelzone' + str(div))
                water_cell=openmc.Cell(fill=self.mat_dict['coolant'][div], region=+Clad_or & +west_boundary & -east_boundary  & +south_boundary & -north_boundary  & +z_planes[div-1] & -top_active, name='waterzone' + str(div))

            else:
                fuel_cell=openmc.Cell(fill=self.mat_dict['fuel'][div], region = -Fuel_or & +z_planes[div-1] & -z_planes[div], name='fuelzone' + str(div))
                water_cell=openmc.Cell(fill=self.mat_dict['coolant'][div], region=+Clad_or & +west_boundary & -east_boundary  & +south_boundary & -north_boundary  & +z_planes[div-1] & -z_planes[div], name='waterzone' + str(div))

            cells_list.append(fuel_cell)
            cells_list.append(water_cell)

        # Adding Clad and Helium which are not updated in temperature
        gap = openmc.Cell(fill=self.mat_dict['non_updated'][2], region=+Fuel_or & -Clad_ir & +bot_active & -top_active)
        clad = openmc.Cell(fill=self.mat_dict['non_updated'][3], region=+Clad_ir & -Clad_or & +bot_active & -top_active)
        clad_top = openmc.Cell(fill=self.mat_dict['non_updated'][3], region= -Clad_or & +top_active & -top_clad)
        clad_bot = openmc.Cell(fill=self.mat_dict['non_updated'][3], region= -Clad_or & +bot_clad & -bot_active)
        fill_water_clad_bot = openmc.Cell(fill=self.mat_dict['non_updated'][4], region=+Clad_or & +west_boundary & -east_boundary  & +south_boundary & -north_boundary  & +bot_clad & -bot_active)
        fill_water_clad_top = openmc.Cell(fill=self.mat_dict['non_updated'][5], region=+Clad_or & +west_boundary & -east_boundary  & +south_boundary & -north_boundary  & +top_active & -top_clad)
        fill_water_bot =  openmc.Cell(fill=self.mat_dict['non_updated'][4], region= +west_boundary & -east_boundary  & +south_boundary & -north_boundary  & +bot_domain & -bot_clad)
        fill_water_top =  openmc.Cell(fill=self.mat_dict['non_updated'][5], region= +west_boundary & -east_boundary  & +south_boundary & -north_boundary  & +top_clad & -top_domain)

        cells_list.append(gap)
        cells_list.append(clad)
        cells_list.append(clad_top)
        cells_list.append(clad_bot)
        cells_list.append(fill_water_clad_top)
        cells_list.append(fill_water_clad_bot)
        cells_list.append(fill_water_top)
        cells_list.append(fill_water_bot)

        self.cells_list = cells_list
    
    def create_settings(self, batches, inactive, s1_val):


        # Indicate how many particles to run
        self.settings = openmc.Settings()
        self.settings.batches = batches
        self.settings.inactive = inactive
        self.settings.particles = s1_val

        # if initialUniformSource:
        #     #Create an initial uniform spatial source distribution over fissionable zones
        #     lower_left = (-square_side/2, -square_side/2, -35)
        #     upper_right = (square_side/2, square_side/2, 35)
        #     uniform_dist = openmc.stats.Box(lower_left, upper_right, only_fissionable=True)
        #     settings.source = openmc.source.Source(space=uniform_dist)

        # if shannonEntropy:
        #     # For source convergence checks, add a mesh that can be used to calculate the Shannon entropy
        #     entropy_mesh = openmc.RegularMesh()
        #     entropy_mesh.lower_left = (-Fuel_or.r, -Fuel_or.r)
        #     entropy_mesh.upper_right = (Fuel_or.r, Fuel_or.r)
        #     entropy_mesh.dimension = (10, 10)
        #     settings.entropy_mesh = entropy_mesh
        
        self.settings.temperature = {'method': 'interpolation'}
    
    def create_tallies(self, tallyDict, meshSize, square_side, l_active, plug_length):

        #### 1 #### SPATIAL TALLY

        # Regular mesh on z direction 
        mesh_z = openmc.RegularMesh()
        mesh_z.dimension = [1, 1, meshSize] #z mesh
        mesh_z.lower_left = [-square_side/2, -square_side/2, -(l_active + 2*plug_length)/2]
        mesh_z.upper_right = [square_side/2, square_side/2, +(l_active + 2*plug_length)/2]

        # Create a mesh filter that can be used in a tally
        mesh_z_filter = openmc.MeshFilter(mesh_z)

        # Now use the mesh filter in a tally and indicate what scores are desired
        mesh_z_tally = openmc.Tally(name=tallyDict['mesh_z'])
        mesh_z_tally.filters = [mesh_z_filter]
        mesh_z_tally.scores = ['flux', 'fission']

        #### 2 #### INTEGRAL TALLY 
        integral_tally = openmc.Tally(name = tallyDict['integral'])
        integral_tally.scores = ['kappa-fission']

        #### 3 #### ENERGY SPECTRUM
        # Let's also create a tally to get the flux energy spectrum. We start by creating an energy filter
        e_min, e_max = 1e-5, 20.0e6
        groups = 100
        energies = np.logspace(log10(e_min), log10(e_max), groups + 1)
        energy_filter = openmc.EnergyFilter(energies)

        spectrum_tally = openmc.Tally(name=tallyDict['spectrum'])
        spectrum_tally.filters = [energy_filter]
        spectrum_tally.scores = ['flux']

        self.tallies = [mesh_z_tally, integral_tally, spectrum_tally]
    
    def create_plots(self):

        path_plots = self.path+'pictures/'

        p_xz = openmc.Plot()
        p_xz.basis = 'xz'
        #p_xz.origin(0.0 , 2.0, 0.0) #plot xz centrato in y=0
        p_xz.filename = path_plots+'pin3D_xz'
        p_xz.width = (2, 400)
        p_xz.pixels = (2000, 2000)
        p_xz.color_by = 'material'


        p_xy = openmc.Plot()
        p_xy.basis = 'xy'
        #p_xy.origin(0.0 , 0.0, 2.0) #plot xy centrato in z=0
        p_xy.filename = path_plots+'pin3D_xy'
        #p_xy.width = (-3, 3)
        p_xy.pixels = (2000, 2000)
        p_xy.color_by = 'material'

        self.plots = openmc.Plots([p_xy, p_xz])

    def export_xml_files(self):

        # Materials
        assert self.mat_dict is not None, 'Materials not created yet'
        
        mat_list = list()
        for key in list(self.mat_dict.keys()):
            for mat in self.mat_dict[key]:
                mat_list.append(mat)
        materials = openmc.Materials(mat_list)
        materials.export_to_xml(self.path+'materials.xml')

        # Geometry
        assert self.cells_list is not None, 'Geometry not created yet'  
        geometry = openmc.Geometry(self.cells_list)
        geometry.export_to_xml(self.path+'geometry.xml')

        # Settings
        assert self.settings is not None, 'Settings not created yet'
        self.settings.export_to_xml(self.path+'settings.xml')

        # Tallies
        assert self.tallies is not None, 'Tallies not created yet'
        tallies = openmc.Tallies(self.tallies)
        tallies.export_to_xml(self.path+'tallies.xml')

        # # Plots
        # assert self.plots is not None, 'Plots not created yet'
        # self.plots.export_to_xml(self.path+'plots.xml')


# ###############################################################################
# # --- MATERIALS 
# from materials import *

# mat_list = list()
# for key in list(mat_dict.keys()):
#     for mat in mat_dict[key]:
#         mat_list.append(mat)
# materials = openmc.Materials(mat_list)
# materials.export_to_xml(path_to_run+'materials.xml')

###############################################################################
# --- GEOMETRY

# # Create cylindrical surfaces
# Fuel_or = openmc.ZCylinder(r=fuel_or, name='Fuel OR')
# Clad_ir = openmc.ZCylinder(r=clad_ir, name='Clad IR')
# Clad_or = openmc.ZCylinder(r=clad_or, name='Clad OR')

# # Create planes for the channel
# square_side = pitch
# east_boundary = openmc.XPlane(x0= square_side/2, boundary_type = 'reflective', name='right boundary')
# west_boundary = openmc.XPlane(x0=-square_side/2, boundary_type = 'reflective', name='left  boundary')
# north_boundary = openmc.YPlane(y0=square_side/2, boundary_type = 'reflective', name='north boundary')
# south_boundary = openmc.YPlane(y0=-square_side/2, boundary_type = 'reflective', name='south boundary')

# z0_bottom = -l_active/2
# z0_top = l_active/2
# dz = (z0_top-z0_bottom)/n_div

# # Create boundaries for the 3D pin
# top_domain = openmc.ZPlane(z0=end_domain_top, boundary_type = 'vacuum', name ='top domain')
# top_clad = openmc.ZPlane(z0=z0_top+plug_length,  name ='top clad')
# top_active= openmc.ZPlane(z0=z0_top, name='top boundary')

# bot_active= openmc.ZPlane(z0=z0_bottom,name='bot boundary')
# bot_clad = openmc.ZPlane(z0=z0_bottom-plug_length, name ='bot clad')
# bot_domain = openmc.ZPlane(z0=end_domain_bot, boundary_type = 'vacuum', name ='bot domain')


# # Create planes for material subdivision
# z_planes = []

# for j in range(n_div-1):#7 piani in mezzo
#     plane = openmc.ZPlane(z0=z0_bottom + dz*(j+1))
#     z_planes.append(plane)


# # Create cells and assigning materials to regions in ascendent order
# cells_list = []
# for div in range(n_div):
#     if div==0: #bottom zone
#         fuel_cell=openmc.Cell(fill=mat_dict['fuel'][div], region = -Fuel_or & +bot_active & -z_planes[div], name='fuelzone' + str(div))
#         water_cell=openmc.Cell(fill=mat_dict['coolant'][div], region=+Clad_or & +west_boundary & -east_boundary  & +south_boundary & -north_boundary  & +bot_active & -z_planes[div], name='waterzone' + str(div))
        
#     elif div==(n_div-1):
#         fuel_cell=openmc.Cell(fill=mat_dict['fuel'][div], region = -Fuel_or & +z_planes[div-1] & -top_active, name='fuelzone' + str(div))
#         water_cell=openmc.Cell(fill=mat_dict['coolant'][div], region=+Clad_or & +west_boundary & -east_boundary  & +south_boundary & -north_boundary  & +z_planes[div-1] & -top_active, name='waterzone' + str(div))

#     else:
#         fuel_cell=openmc.Cell(fill=mat_dict['fuel'][div], region = -Fuel_or & +z_planes[div-1] & -z_planes[div], name='fuelzone' + str(div))
#         water_cell=openmc.Cell(fill=mat_dict['coolant'][div], region=+Clad_or & +west_boundary & -east_boundary  & +south_boundary & -north_boundary  & +z_planes[div-1] & -z_planes[div], name='waterzone' + str(div))

#     cells_list.append(fuel_cell)
#     cells_list.append(water_cell)



# # Adding Clad and Helium which are not updated in temperature
# gap = openmc.Cell(fill=mat_dict['non_updated'][2], region=+Fuel_or & -Clad_ir & +bot_active & -top_active)
# clad = openmc.Cell(fill=mat_dict['non_updated'][3], region=+Clad_ir & -Clad_or & +bot_active & -top_active)
# clad_top = openmc.Cell(fill=mat_dict['non_updated'][3], region= -Clad_or & +top_active & -top_clad)
# clad_bot = openmc.Cell(fill=mat_dict['non_updated'][3], region= -Clad_or & +bot_clad & -bot_active)
# fill_water_clad_bot = openmc.Cell(fill=mat_dict['non_updated'][4], region=+Clad_or & +west_boundary & -east_boundary  & +south_boundary & -north_boundary  & +bot_clad & -bot_active)
# fill_water_clad_top = openmc.Cell(fill=mat_dict['non_updated'][5], region=+Clad_or & +west_boundary & -east_boundary  & +south_boundary & -north_boundary  & +top_active & -top_clad)
# fill_water_bot =  openmc.Cell(fill=mat_dict['non_updated'][4], region= +west_boundary & -east_boundary  & +south_boundary & -north_boundary  & +bot_domain & -bot_clad)
# fill_water_top =  openmc.Cell(fill=mat_dict['non_updated'][5], region= +west_boundary & -east_boundary  & +south_boundary & -north_boundary  & +top_clad & -top_domain)

# cells_list.append(gap)
# cells_list.append(clad)
# cells_list.append(clad_top)
# cells_list.append(clad_bot)
# cells_list.append(fill_water_clad_top)
# cells_list.append(fill_water_clad_bot)
# cells_list.append(fill_water_top)
# cells_list.append(fill_water_bot)

# # Create a geometry and export to XML
# geometry = openmc.Geometry(cells_list)
# geometry.export_to_xml(path_to_run+'geometry.xml')

###############################################################################
# --- SETTINGS

# # Indicate how many particles to run
# settings = openmc.Settings()
# settings.batches = batches
# settings.inactive = inactive
# settings.particles = s1_val

# if initialUniformSource:
#     #Create an initial uniform spatial source distribution over fissionable zones
#     lower_left = (-square_side/2, -square_side/2, -35)
#     upper_right = (square_side/2, square_side/2, 35)
#     uniform_dist = openmc.stats.Box(lower_left, upper_right, only_fissionable=True)
#     settings.source = openmc.source.Source(space=uniform_dist)

# if shannonEntropy:
#     # For source convergence checks, add a mesh that can be used to calculate the Shannon entropy
#     entropy_mesh = openmc.RegularMesh()
#     entropy_mesh.lower_left = (-Fuel_or.r, -Fuel_or.r)
#     entropy_mesh.upper_right = (Fuel_or.r, Fuel_or.r)
#     entropy_mesh.dimension = (10, 10)
#     settings.entropy_mesh = entropy_mesh

# settings.temperature = {'method': 'interpolation'}
# settings.export_to_xml(path_to_run+'settings.xml')

################################################################################
# --- TALLIES

# #### 1 #### SPATIAL TALLY

# # Regular mesh on z direction 
# mesh_z = openmc.RegularMesh()
# mesh_z.dimension = [1, 1, meshSize] #z mesh
# mesh_z.lower_left = [-square_side/2, -square_side/2, -(l_active + 2*plug_length)/2]
# mesh_z.upper_right = [square_side/2, square_side/2, +(l_active + 2*plug_length)/2]

# # Create a mesh filter that can be used in a tally
# mesh_z_filter = openmc.MeshFilter(mesh_z)

# # Now use the mesh filter in a tally and indicate what scores are desired
# mesh_z_tally = openmc.Tally(name=tallyDict['mesh_z'])
# mesh_z_tally.filters = [mesh_z_filter]
# mesh_z_tally.scores = ['flux', 'fission']

# #### 2 #### INTEGRAL TALLY 
# integral_tally = openmc.Tally(name = tallyDict['integral'])
# integral_tally.scores = ['kappa-fission']

# #### 3 #### ENERGY SPECTRUM
# # Let's also create a tally to get the flux energy spectrum. We start by creating an energy filter
# e_min, e_max = 1e-5, 20.0e6
# groups = 100
# energies = np.logspace(log10(e_min), log10(e_max), groups + 1)
# energy_filter = openmc.EnergyFilter(energies)

# spectrum_tally = openmc.Tally(name=tallyDict['spectrum'])
# spectrum_tally.filters = [energy_filter]
# spectrum_tally.scores = ['flux']

# # Instantiate a Tallies collection and export to XML
# tallies = openmc.Tallies([mesh_z_tally, integral_tally, spectrum_tally])
# tallies.export_to_xml(path_to_run+'tallies.xml')

###############################################################################
# --- PLOT

# p_xz = openmc.Plot()
# p_xz.basis = 'xz'
# #p_xz.origin(0.0 , 2.0, 0.0) #plot xz centrato in y=0
# p_xz.filename = './pictures/pin3D_xz'
# p_xz.width = (2, 400)
# p_xz.pixels = (2000, 2000)
# p_xz.color_by = 'material'


# p_xy = openmc.Plot()
# p_xy.basis = 'xy'
# #p_xy.origin(0.0 , 0.0, 2.0) #plot xy centrato in z=0
# p_xy.filename = './pictures/pin3D_xy'
# #p_xy.width = (-3, 3)
# p_xy.pixels = (2000, 2000)
# p_xy.color_by = 'material'

# plots = openmc.Plots([p_xy, p_xz])
# plots.export_to_xml(path_to_run+'plots.xml')

# openmc.plot_geometry()
