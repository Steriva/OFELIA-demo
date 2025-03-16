import os
import openmc
import numpy as np

class PicardSteps():
    def __init__(self, path, max_iter, tol, batches, s1):
        self.max_iter = max_iter
        self.tol = tol
        self.path = path

        self.batch = batches
        self.s1 = s1

        self.iter = 0

    def run_openmc(self, threads = 8, output=False):

        input_folder = self.path + '/build_xml/it' + str(self.iter)

        os.chdir(input_folder)
        print('  Running OpenMC')
        openmc.run(threads = threads, cwd=input_folder, output = output)
        os.chdir(self.path)

        # Move statepoints
        old_sp_name = 'statepoint.'+str(self.batch)+'.h5'
        old_sp_path = input_folder + '/' +old_sp_name

        new_sp_name = 'statepoint.'+str(self.batch)+'.it' + str(self.iter) + '.h5'
        new_sp_path = input_folder + '/' +new_sp_name

        os.system('mv ' + old_sp_path + ' ' + new_sp_path)

        return openmc.StatePoint(new_sp_path)

    def under_relaxation(self, sum_s, q3_unrelaxed, q3_relaxed_list):

        # under-relaxation and population treatment
        if self.iter == 0:  
            
            q3_relaxed = q3_unrelaxed

            # Under Relaxation
            sn = (self.s1+ np.sqrt(self.s1**2 + 4 * self.s1 * sum_s))/2
            sum_s=sn
            alpha = sn/sum_s
            
        if self.iter>0:
            sn = (self.s1+ np.sqrt(self.s1**2 + 4 * self.s1 * sum_s))/2
            sum_s = sum_s + sn
            alpha = sn/sum_s

        
            q3_relaxed = q3_unrelaxed * alpha + q3_relaxed_list[self.iter-1] * (1-alpha)

        return sn, sum_s, q3_relaxed, alpha
            