import uuid
import os
from subprocess import Popen, PIPE

class Config:
    def __init__(self, nproc):
        self.scratch = '.'
        self.submitdir = '.'
        self.batch = True
        self.procs = []
        # self.mpi_not_setup = True
        if 'ESP_PSP_PATH' not in os.environ:
            os.environ['ESP_PSP_PATH'] = '.'
        self.nproc = nproc
        self.perHostMpiExec =  'mpiexec --use-hwthread-cpus -np '+str(self.nproc)
        self.perProcMpiExec =  'mpiexec --use-hwthread-cpus -np '+str(self.nproc)+' -wdir %s %s'
        self.perSpecProcMpiExec = 'mpiexec --use-hwthread-cpus -np '+str(self.nproc)+' -wdir %s %s'
        # assign a random job id
        self.jobid = uuid.uuid4().hex[:8]

    def do_perProcMpiExec(self, workdir, program):
        execute = Popen(self.perProcMpiExec % (workdir, program), shell=True, stdin=PIPE, stdout=PIPE)
        return (execute.stdin, execute.stdout)

    def do_perProcMpiExec_outputonly(self, workdir, program):
        return Popen(self.perProcMpiExec % (workdir, program), shell=True, stdout=PIPE).stdout

    def runonly_perProcMpiExec(self, workdir, program):
        os.system(self.perProcMpiExec % (workdir, program))

    def do_perSpecProcMpiExec(self, machinefile, nproc, workdir, program):
        execute = Popen(self.perProcMpiExec % (workdir, program), shell=True, stdin=PIPE, stdout=PIPE, stderr=PIPE)
        return (execute.stdin, execute.stdout, execute.stderr)
