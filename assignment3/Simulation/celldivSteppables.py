from cc3d.core.PySteppables import *
import numpy as np
from fipy import CellVariable, Grid2D, TransientTerm, DiffusionTerm, Viewer, ImplicitSourceTerm#you can install fipy and import fipy, the finite volume solver here to solve your reaction diffusion equations and couple them the concentrations to interact with cells



#I am defining my finite volume mesh here****************
dx = 1.
dy = 1.
global nx # I am getting my CPM grid size from the xml file <Dimensions x="100" y="100" z="1"/>
global ny
nx = 100
ny =100
mesh = Grid2D(dx=dx, dy=dy,nx =nx, ny=ny ) 
# I am setting my diffusion coeffecient here
D_metabolite1 = 0.1
D_metabolite2 = 0.01
D = max(D_metabolite1, D_metabolite2) #largest diffusion coeffecient in my system to determine the least delta t
# I am using Fourier number to find time step for my finite volume solver
delta_t = 0.5 * (1/(D*(1/dx**2 + 1/dy**2)))
print (delta_t)
#Setting inital concentration of metabolites
metabolite1 = CellVariable(name="Metabolite 1", mesh=mesh, value=10.0) # in this example this metabolite1 is consumed by cells
metabolite2 = CellVariable(name="Metabolite 2", mesh=mesh, value=0.0) # in this example this metabolite2 is secreted by cells
CellPresent = CellVariable(name="Cell presence array", mesh=mesh, value=0.0) # this variable stores where my cells are
#Stating boundary condition - setting Dirichlet of 5 conc along x = 0 and Neumann dc/dn= 0 along x=nx and y boundaries (change as you require)
metabolite1.constrain([10.], mesh.facesLeft) # Dirichlet
metabolite1.faceGrad.constrain([0.], mesh.facesRight) #Neumann
metabolite1.faceGrad.constrain([0.], mesh.facesTop)
metabolite1.faceGrad.constrain([0.], mesh.facesBottom)
metabolite2.faceGrad.constrain([0.], mesh.facesLeft) # 
metabolite2.faceGrad.constrain([0.], mesh.facesRight) #Neumann
metabolite2.faceGrad.constrain([0.], mesh.facesTop)
metabolite2.faceGrad.constrain([0.], mesh.facesBottom)
#*****************End Fipy mesh and boundary definitions *******************************

 # I will initialize the position of my cells in my simulation here!
class ConstraintInitializerSteppable(SteppableBasePy):
    def __init__(self,frequency=1):
        SteppableBasePy.__init__(self,frequency)

    def start(self):  
        # I am creating my cells of size 5*5 here individually (three cells), defining their x,y coordinates and cell type
        self.cell_field[10:15, 50:55, 0] = self.new_cell(self.CONDENSING)
        self.cell_field[40:45, 40:45, 0] = self.new_cell(self.CONDENSING)
        self.cell_field[60:65, 60:65, 0] = self.new_cell(self.NONCONDENSING)        
        for cell in self.cell_list:
            cell.targetVolume = 25
            cell.lambdaVolume = 2.0
        
        
class GrowthSteppable(SteppableBasePy):
    def __init__(self,frequency=1):
        SteppableBasePy.__init__(self, frequency)

    # This function runs every step or mcs!
    def step(self, mcs):
        def coordinate_converter(coord):
            global nx, ny
            # this function just converts x,y,z coordinate from cc3d to a single index for fipy
            ind = coord[0]+coord[1]*nx #if you have three dimensions fix this!!! with +coord[2]*nx*ny
            return ind
            
        # I am assuming my cells grow only once in 60 mcs = 25 times steps (could be seconds, minutes or hours based on metabolite1 diffusivity units!)
        if mcs%60==0:
            
            for cell in self.cell_list:
                # Make sure PixelTracker plugin is loaded
                pixel_list = self.get_cell_pixel_list(cell)
                for pixel_tracker_data in pixel_list:
                    pos = [pixel_tracker_data.pixel.x, pixel_tracker_data.pixel.y, pixel_tracker_data.pixel.z]
                    ind = coordinate_converter(pos)
                    CellPresent[ind]=1
            time  = 25 # takes the time units of your diffusion coeffecient
            mumax = 0.5 #maximum growth rate
            Ks = 4 # half saturation coeffecient
            Y = 0.8 # Yield coeffecient
            ksec = 0.0005
            eq1 = TransientTerm(var=metabolite1) == DiffusionTerm(coeff=D_metabolite1, var=metabolite1) - ImplicitSourceTerm(mumax/Y*CellPresent/(Ks+metabolite1), var = metabolite1) #Implicit source term includes a Concentration term of metabolite
            eq2 = TransientTerm(var=metabolite2) == DiffusionTerm(coeff=D_metabolite2, var=metabolite2) + ksec*CellPresent
            eq = eq1 & eq2
            for step in range(int(time/delta_t)):
                eq.solve(dt=delta_t)
            # Preparing to transfer from FiPy to CC3D array
            field = self.field.METABOLITE1
            metabolite_reshape  = metabolite1.value.reshape(nx,ny,1, order='F')
            field2 = self.field.METABOLITE2
            metabolite_reshape2  = metabolite2.value.reshape(nx,ny,1, order='F')
            if field:
                for i, j, k in self.every_pixel():
                    field[i,j,k] =  metabolite_reshape[i,j,k] # I am copying from Fipy to cc3d field value for visualization purposes
                    field2[i,j,k] =  metabolite_reshape2[i,j,k]
            for cell in self.cell_list:
                COM_POS = [int(cell.xCOM), int(cell.yCOM), int(cell.zCOM)]
                ind = coordinate_converter(COM_POS)
                concentrationAtCOM = metabolite1[ind] # Find the field value at center of the cell
                # you could just keep adding target volume regulary saying your cell consistently grows (maximum nutrient concentration always available)by uncommenting line below
                # cell.targetVolume += 0     
                # alternatively if you want to make growth function from chemical concentration at center of mass use the lines below  
                cell.targetVolume += float(mumax*cell.volume*concentrationAtCOM/(Ks+concentrationAtCOM)) # I am just assuming volume = mass so density = 1, also you will need to multiply by proper dt here
            CellPresent[:]=0 # reset cell presence array for next iteration

        
class MitosisSteppable(MitosisSteppableBase):
    def __init__(self,frequency=1):
        MitosisSteppableBase.__init__(self,frequency)

    def step(self, mcs):

        cells_to_divide=[]
        for cell in self.cell_list:
            if cell.volume>50: #Divide if volume is greater than 50
                cells_to_divide.append(cell)

        for cell in cells_to_divide:

            self.divide_cell_random_orientation(cell)
            # Other valid options
            # self.divide_cell_orientation_vector_based(cell,1,1,0)
            # self.divide_cell_along_major_axis(cell)
            # self.divide_cell_along_minor_axis(cell)

    def update_attributes(self):
        # reducing parent target volume
        self.parent_cell.targetVolume /= 2.0                  

        self.clone_parent_2_child()            

        # for more control of what gets copied from parent to child use cloneAttributes function
        # self.clone_attributes(source_cell=self.parent_cell, target_cell=self.child_cell, no_clone_key_dict_list=[attrib1, attrib2]) 
        
        if self.parent_cell.type==1:
            self.child_cell.type=1
        else:
            self.child_cell.type=2

        