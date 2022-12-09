# This is a placeholder espsite.py
# Replace this with one of the example files and adjust
# to your cluster.
import os


class Config:
    def __init__(self):
        self.scratch = '.'
        self.submitdir = '.'
        self.batch = False
        # self.mpi_not_setup = True
        if 'ESP_PSP_PATH' not in os.environ:
            os.environ['ESP_PSP_PATH'] = '.'

    def do_perProcMpiExec(self, workdir, program):
        return None

    def do_perProcMpiExec_outputonly(self, workdir, program):
        return None

    def runonly_perProcMpiExec(self, workdir, program):
        pass

    def do_perSpecProcMpiExec(self, machinefile, nproc, workdir, program):
        return None
