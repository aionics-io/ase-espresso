from . import utils
import numpy as np
from ase.units import Rydberg, Bohr
from subprocess import Popen, PIPE, call, check_output
import tarfile
import os


class Mixins:

    def write_input(self):

        f = open('input.pwi', 'w')

        ionssec = self.calculation not in ('scf', 'nscf', 'bands')

        # &CONTROL ###
        efield = (self.field['status'])
        dipfield = (self.dipole['status'])

        # We basically ignore convergence of total energy differences
        # between ionic steps and only consider fmax as in ase
        print('  etot_conv_thr=1d0,', file=f)
        self.forc_conv_thr /= Rydberg / Bohr
        print('  forc_conv_thr={},'.format(self.forc_conv_thr), file=f)

        if self.fix_magmom:
            assert self.spinpol
            self.totmag = self.summed_magmoms
            print('  tot_magnetization=' + utils.num2str(self.totmag) + ',', file=f)
        elif self.tot_magnetization != -1:
            self.totmag = self.tot_magnetization
            print('  tot_magnetization=' + utils.num2str(self.totmag) + ',', file=f)

        if self.ecutfock is not None:
            print('  ecutfock=' + utils.num2str(self.ecutfock / Rydberg) + ',', file=f)

        if self.nbnd is not None:
            # set number of bands
            if self.nbnd > 0:
                self.nbnd = int(self.nbnd)
            else:
                # if self.nbnd is negative create -self.nbnd extra bands
                if self.nvalence is None:
                    self.nvalence, self.nel = self.get_nvalence()
                if self.noncollinear:
                    self.nbnd = int(np.sum(self.nvalence) - self.nbnd * 2.)
                else:
                    self.nbnd = int(np.sum(self.nvalence) / 2. - self.nbnd)
            print('  nbnd=' + str(self.nbnd) + ',', file=f)

        if usetetrahedra:
            print('  occupations=\'tetrahedra\',', file=f)
        else:
            if abs(self.sigma) > 1e-13:
                print('  occupations=\'' + self.occupations + '\',', file=f)
                print('  smearing=\'' + self.smearing + '\',', file=f)
                print(
                    '  degauss=' + utils.num2str(self.sigma / Rydberg) + ',',
                    file=f)
            else:
                if self.spinpol:
                    assert self.fix_magmom
                print('  occupations=\'fixed\',', file=f)

        if self.spinpol:
            print('  nspin=2,', file=f)
            spcount = 1
            if self.nel is None:
                self.nvalence, self.nel = self.get_nvalence()
                # FOLLOW SAME ORDERING ROUTINE AS FOR PSP
            for species in self.species:
                spec = self.specdict[species]
                el = spec.s
                mag = spec.magmom / self.nel[el]
                assert np.abs(mag) <= 1.  # magnetization oversaturated!!!
                print('  starting_magnetization(%d)=%s,' % (spcount, utils.num2str(float(mag))), file=f)
                spcount += 1

        elif self.noncollinear:
            print('  noncolin=.true.,', file=f)
            if self.spinorbit:
                print('  lspinorb=.true.', file=f)
            spcount = 1
            if self.nel is None:
                self.nvalence, self.nel = self.get_nvalence()

            for species in self.species:
                spec = self.specdict[species]
                el = spec.s
                mag = spec.magmom / self.nel[el]
                assert np.abs(mag) <= 1.  # magnetization oversaturated!!!
                print(
                    '  starting_magnetization(%d)=%s,' %
                    (spcount, utils.num2str(float(mag))),
                    file=f)
                spcount += 1

        if self.isolated is not None:
            print('  assume_isolated=\'' + self.isolated + '\',', file=f)
        print('  input_dft=\'' + self.xc + '\',', file=f)

        if self.beefensemble:
            print('  ensemble_energies=.true.,', file=f)
            if self.printensemble:
                print('  print_ensemble_energies=.true.,', file=f)
            else:
                print('  print_ensemble_energies=.false.,', file=f)

        if dipfield:
            try:
                edir = self.dipole['edir']
            except BaseException:
                pass
        elif efield:
            try:
                edir = self.field['edir']
            except BaseException:
                pass

        if dipfield or efield:
            print('  edir=' + str(edir) + ',', file=f)

        if dipfield:
            if 'emaxpos' in self.dipole:
                emaxpos = self.dipole['emaxpos']
            else:
                emaxpos = self.find_max_empty_space(edir)
            if 'eopreg' in self.dipole:
                eopreg = self.dipole['eopreg']
            else:
                eopreg = 0.025
            if 'eamp' in self.dipole:
                eamp = self.dipole['eamp']
            else:
                eamp = 0.0
            print('  emaxpos=' + utils.num2str(emaxpos) + ',', file=f)
            print('  eopreg=' + utils.num2str(eopreg) + ',', file=f)
            print('  eamp=' + utils.num2str(eamp) + ',', file=f)
        if efield:
            if 'emaxpos' in self.field:
                emaxpos = self.field['emaxpos']
            else:
                emaxpos = 0.0
            if 'eopreg' in self.field:
                eopreg = self.field['eopreg']
            else:
                eopreg = 0.0
            if 'eamp' in self.field:
                eamp = self.field['eamp']
            else:
                eamp = 0.0
            print('  emaxpos=' + utils.num2str(emaxpos) + ',', file=f)
            print('  eopreg=' + utils.num2str(eopreg) + ',', file=f)
            print('  eamp=' + utils.num2str(eamp) + ',', file=f)

        # &ELECTRONS ###
        print('/\n&ELECTRONS', file=f)

        self.conv_thr /= Rydberg
        print('  conv_thr=' + utils.num2str(self.conv_thr) + ',', file=f)

        # &IONS ###
        if not ionssec:
            simpleconstr, otherconstr = [], []
        else:
            simpleconstr, otherconstr = utils.convert_constraints(self.atoms)

        if self.ion_dynamics is None:
            self.optdamp = False
        else:
            self.optdamp = (self.ion_dynamics.upper() == 'DAMP')

        if self.ion_dynamics is not None and ionssec:
            if len(otherconstr) != 0:
                print('/\n&IONS\n  ion_dynamics=\'damp\',', file=f)
                self.optdamp = True
            else:
                print('/\n&IONS\n  ion_dynamics=\'' + self.ion_dynamics + '\',', file=f)
            if self.ion_positions is not None:
                print('  ion_positions=\'' + self.ion_positions + '\',', file=f)
        elif self.ion_positions is not None:
            print('/\n&IONS\n  ion_positions=\'' + self.ion_positions + '\',', file=f)

        # &CELL ###
        if self.cell_dynamics is not None:
            print('/\n&CELL\n  cell_dynamics=\'' + self.cell_dynamics + '\',', file=f)
            if self.press is not None:
                print('  press=' + utils.num2str(self.press) + ',', file=f)
            if self.dpress is not None:
                print('  press_conv_thr=' + utils.num2str(self.dpress) + ',', file=f)
            if self.cell_factor is not None:
                print('  cell_factor=' + utils.num2str(self.cell_factor) + ',', file=f)
            if self.cell_dofree is not None:
                print('  cell_dofree=\'' + self.cell_dofree + '\',', file=f)

        f.close()

    def read(self, atoms):

        if not self.started:
            fresh = True
            self.initialize(atoms)
        else:
            fresh = False

        if self.recalculate:
            if not fresh:
                self.cinp.flush()
            s = open(self.log, 'a')
            a = self.cout.readline().decode('utf-8')
            s.write(a)
            atom_occ = {}
            magmoms = np.zeros(len(atoms))
            while (a != '' and
                   a[:17] != '!    total energy' and
                   a[:13] != '     stopping' and
                   a[:20] != '     convergence NOT'):
                a = self.cout.readline().decode('utf-8')
                s.write(a)
                s.flush()

                if a[:19] == '     iteration #  1':
                    while (a != '' and a[:17] != '!    total energy'
                           and a[:13] != '     stopping'
                           and a[:20] != '     convergence NOT'
                           and a[:22] != ' --- exit write_ns ---'):
                        a = self.cout.readline().decode('utf-8')
                        s.write(a)
                        s.flush()
                        if a[:5] == 'atom ':
                            atomnum = int(a[8:10])
                            # 'atom    1   Tr[ns(na)] =   1.00000'
                            if a[12:25] == 'Tr[ns(na)] = ':
                                N0 = float(a[27:35]) / 2.
                            elif a[12:42] == 'Tr[ns(na)] (up, down, total) =':
                                N0 = [
                                    float(a[42:52]),
                                    float(a[53:62]),
                                    float(a[63:71])
                                ]
                                N0 = N0[-1]  # only taking the total occupation
                            atom_occ[atomnum - 1] = {}
                            atom_occ[atomnum - 1][0] = N0
                if a[:39] == '     End of self-consistent calculation':
                    while (a != '' and
                           a[:17] != '!    total energy' and
                           a[:13] != '     stopping' and
                           a[:20] != '     convergence NOT'):
                        a = self.cout.readline().decode('utf-8')
                        s.write(a)
                        s.flush()
                        if a[:5] == 'atom ':
                            atomnum = int(a[8:10])
                            if a[12:25] == 'Tr[ns(na)] = ':
                                Nks = float(a[27:35]) / 2.
                            elif a[12:42] == 'Tr[ns(na)] (up, down, total) =':
                                Nks = [
                                    float(a[42:52]),
                                    float(a[53:62]),
                                    float(a[63:71])
                                ]
                                # only taking the total occupation
                                Nks = Nks[-1]
                                magmom = Nks[0] - Nks[1]
                                magmoms[atomnum] = magmom
                            atom_occ[atomnum - 1]['ks'] = Nks
                    break
            if a[:20] == '     convergence NOT':
                self.stop()
                raise KohnShamConvergenceError(
                    'scf cycles did not converge\nincrease maximum '
                    'number of steps and/or decreasing mixing'
                )
            elif a[:13] == '     stopping':
                self.stop()
                self.checkerror()
                # if checkerror shouldn't find an error here,
                # throw this generic error
                raise RuntimeError('SCF calculation failed')
            elif a == '' and self.calculation in ('relax', 'scf', 'vc-relax',
                                                  'vc-md', 'md'):
                self.checkerror()
                # if checkerror shouldn't find an error here,
                # throw this generic error
                raise RuntimeError('SCF calculation failed')
            self.atom_occ = atom_occ
            self.results['magmoms'] = magmoms
            self.results['magmom'] = np.sum(magmoms)
            if self.calculation in ('relax', 'scf', 'vc-relax', 'vc-md', 'md'):
                self.energy_free = float(a.split()[-2]) * Rydberg
                # get S*T correction (there is none for Marzari-Vanderbilt=Cold
                # smearing)
                if (self.occupations == 'smearing' and
                    self.smearing[0].upper() != 'M' and
                    self.smearing[0].upper() != 'C' and
                    not self.optdamp):
                    a = self.cout.readline().decode('utf-8')
                    s.write(a)
                    exx = False
                    while a[:13] != '     smearing':
                        a = self.cout.readline().decode('utf-8')
                        s.write(a)
                        if a.find('EXX') > -1:
                            exx = True
                            break
                    if exx:
                        self.ST = 0.0
                        self.energy_zero = self.energy_free
                    else:
                        self.ST = -float(a.split()[-2]) * Rydberg
                        self.energy_zero = self.energy_free + 0.5 * self.ST
                else:
                    self.ST = 0.0
                    self.energy_zero = self.energy_free
            else:
                self.energy_free = None
                self.energy_zero = None

            self.got_energy = True
            self.results['energy'] = self.energy_zero
            self.results['free_energy'] = self.energy_free

            a = self.cout.readline().decode('utf-8')
            s.write(a)
            s.flush()

            if self.calculation in ('relax', 'scf', 'vc-relax', 'vc-md', 'md'):
                a = self.cout.readline().decode('utf-8')
                s.write(a)
                if not self.dontcalcforces:
                    while a[:11] != '     Forces':
                        a = self.cout.readline().decode('utf-8')
                        s.write(a)
                        s.flush()
                    a = self.cout.readline().decode('utf-8')
                    s.write(a)
                    self.forces = np.empty((self.natoms, 3), np.float)
                    for i in range(self.natoms):
                        a = self.cout.readline().decode('utf-8')
                        while a.find('force') < 0:
                            s.write(a)
                            a = self.cout.readline().decode('utf-8')
                        s.write(a)
                        forceinp = a.split()
                        self.forces[i][:] = [
                            float(x) for x in forceinp[len(forceinp) - 3:]
                        ]
                    self.forces *= (Rydberg / Bohr)
                else:
                    self.forces = None
            else:
                self.forces = None
            self.recalculate = False
            s.close()
            self.results['forces'] = self.forces
            self.stop()

            # get final energy and forces for internal QE relaxation run
            if self.calculation in ('relax', 'vc-relax', 'vc-md', 'md'):
                p = Popen(
                    'grep -a -n "!    total" ' + self.log + ' | tail -1',
                    shell=True, stdout=PIPE).stdout
                n = int(p.readline().decode('utf-8').split(':')[0]) - 1
                f = open(self.log, 'r')
                for i in range(n):
                    f.readline()
                self.energy_free = float(f.readline().split()[-2]) * Rydberg
                # get S*T correction (there is none for Marzari-Vanderbilt=Cold
                # smearing)
                if (self.occupations == 'smearing' and
                    self.smearing[0].upper() != 'M' and
                    self.smearing[0].upper() != 'C' and
                    not self.optdamp):
                    a = f.readline()
                    exx = False
                    while a[:13] != '     smearing':
                        a = f.readline()
                        if a.find('EXX') > -1:
                            exx = True
                            break
                    if exx:
                        self.ST = 0.0
                        self.energy_zero = self.energy_free
                    else:
                        self.ST = -float(a.split()[-2]) * Rydberg
                        self.energy_zero = self.energy_free + 0.5 * self.ST
                else:
                    self.ST = 0.0
                    self.energy_zero = self.energy_free

                if (self.U_projection_type == 'atomic' and not
                    self.dontcalcforces):
                    a = f.readline()
                    while a[:11] != '     Forces':
                        a = f.readline()
                    f.readline()
                    self.forces = np.empty((self.natoms, 3), np.float)
                    for i in range(self.natoms):
                        a = f.readline()
                        while a.find('force') < 0:
                            a = f.readline()
                        forceinp = a.split()
                        self.forces[i][:] = [
                            float(x) for x in forceinp[len(forceinp) - 3:]
                        ]
                    self.forces *= (Rydberg / Bohr)
                f.close()

            self.checkerror()

    def initialize(self, atoms):
        """Create the input.pwi input file and start the calculation."""
        if not self.started:
            stdout = check_output(['which', 'pw.x']).decode()
            self.exedir = os.path.dirname(stdout)

            with open(self.log, 'a') as f:
                f.write('  python dir          : {}\n'.format(self.mypath))
                f.write('  espresso dir        : {}\n'.format(self.exedir))
                f.write('  pseudo dir          : {}\n'.format(self.psppath))

            self.natoms = len(self.atoms)
            self.check_spinpol()
            self.write_input()

        self.start()

    def start(self):
        if not self.started:
            if self.single_calculator:
                while len(self.calculators) > 0:
                    self.calculators.pop().stop()
                self.calculators.append(self)
            if self.site.batch:
                cdir = os.getcwd()
                os.chdir(self.localtmp)
                call(self.site.perHostMpiExec + ' cp ' + self.localtmp +
                     '/input.pwi ' + self.scratch, shell=True)

                if not self.proclist:
                    self.cinp, self.cout = self.site.do_perProcMpiExec(
                        self.scratch, self.exedir + 'pw.x ' +
                        self.parflags + ' -in input.pwi')
                else:
                    (self.cinp,
                     self.cout,
                     self.cerr) = self.site.do_perSpecProcMpiExec(
                        self.mycpus, self.myncpus, self.scratch,
                        self.exedir + 'pw.x ' + self.parflags +
                        ' -in input.pwi|' + self.mypath + '/espfilter ' + str(
                            self.natoms) + ' ' + self.log + '0')
                os.chdir(cdir)
            else:
                call('cp ' + self.localtmp + '/input.pwi ' + self.scratch,
                     shell=True)
                cmd = 'cd ' + self.scratch + ' ; ' + self.exedir + 'pw.x ' + self.serflags + ' -in input.pwi'
                p = Popen(cmd, shell=True, stdin=PIPE,
                          stdout=PIPE, close_fds=True)
                self.cinp, self.cout = (p.stdin, p.stdout)

            self.started = True

    def stop(self):
        if self.started:
            self.cinp.flush()
            s = open(self.log, 'a')
            a = self.cout.readline().decode('utf-8')
            s.write(a)
            while a != '':
                a = self.cout.readline().decode('utf-8')
                s.write(a)
                s.flush()
            s.close()
            self.cinp.close()
            self.cout.close()
            self.started = False

    def topath(self, filename):
        if os.path.isabs(filename):
            return filename
        else:
            return os.path.join(self.sdir, filename)

    def get_nonselfconsistent_energies(self, type='beefvdw'):
        self.stop()
        cmd = 'grep -a -32 "BEEF-vdW xc energy contributions" ' + self.log + ' | tail -32'
        p = Popen(cmd, shell=True, stdout=PIPE).stdout
        s = p.readlines()
        xc = np.array([])
        for i, l in enumerate(s):
            l_ = float(l.split(":")[-1]) * Rydberg
            xc = np.append(xc, l_)
        assert len(xc) == 32
        return xc

    def find_max_empty_space(self, edir=3):
        """Assuming periodic boundary conditions, finds the largest
        continuous segment of free, unoccupied space and returns
        its midpoint in scaled coordinates (0 to 1) in the edir
        direction (default z).
        """
        position_array = self.atoms.get_scaled_positions()[..., edir - 1]
        position_array.sort()
        differences = np.diff(position_array)
        differences = np.append(differences, position_array[0] + 1 - position_array[-1])
        max_diff_index = np.argmax(differences)
        if max_diff_index == len(position_array) - 1:
            return (position_array[0] + 1 + position_array[-1]) / 2. % 1
        else:
            return (position_array[max_diff_index] + position_array[max_diff_index + 1]) / 2.

    def checkerror(self):
        cmd = 'grep -a -n Giannozzi ' + self.log + ' | tail -1'
        p = Popen(cmd, shell=True, stdout=PIPE).stdout
        try:
            out = p.readline().decode('utf-8')
            n = int(out.split()[0].strip(':'))
        except BaseException:
            raise RuntimeError(
                'Espresso executable doesn\'t seem to have been started.')

        cmd = ('tail -n +%d ' % n) + self.log + ' | grep -a -n %%%%%%%%%%%%%%%% |tail -2'
        p = Popen(cmd, shell=True, stdout=PIPE).stdout
        s = p.readlines()

        if len(s) < 2:
            return

        a = int(s[0].decode('utf-8').split()[0].strip(':')) + 1
        b = int(s[1].decode('utf-8').split()[0].strip(':')) - a

        if b < 1:
            return

        cmd = ('tail -n +%d ' % (a + n - 1)) + self.log + ('|head -%d' % b)
        p = Popen(cmd, shell=True, stdout=PIPE).stdout
        err = p.readlines()

        if err[0].decode('utf-8').lower().find('error') < 0:
            return

        msg = ''
        for e in err:
            msg += e.decode('utf-8')
        raise RuntimeError(msg[:len(msg) - 1])

