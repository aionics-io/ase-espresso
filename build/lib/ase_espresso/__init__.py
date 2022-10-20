#****************************************************************************
# Copyright (C) 2013-2015 SUNCAT
# This file is distributed under the terms of the
# GNU General Public License. See the file `COPYING'
# in the root directory of the present distribution,
# or http://www.gnu.org/copyleft/gpl.txt .
#****************************************************************************

import os
from subprocess import Popen, PIPE, call
import multiprocessing
import warnings
import atexit
import sys
from io import BytesIO

import numpy as np
from ase import Atoms
from ase.units import Rydberg, Bohr, Hartree
from ase.calculators.calculator import kptdensity2monkhorstpack

from .atomic_configs import hundmag
from .worldstub import world
from . import subdirs
from . import utils

try:
    from ase.calculators.calculator import FileIOCalculator as Calculator
except BaseException:
    from ase.calculators.general import Calculator

try:
    from . import espsite
except ImportError:
    print(
        '*** ase-espresso requires a site-specific espsite.py in PYTHONPATH. '
        '*** You may use the espsite.py.example.* in the git checkout as templates.'
    )
    raise ImportError
gitver = 'GITVERSION'

# ase controlled pw.x's register themselves here, so they can be
# stopped automatically
espresso_calculators = []

# Define types of convergence errors that can be used to handle
# convergence error automatically


class ConvergenceError(Exception):
    pass


class KohnShamConvergenceError(ConvergenceError):
    pass

pkgpath = os.path.abspath(os.path.dirname(__file__))
rootpath = os.path.dirname(pkgpath)

class espresso(Calculator):
    """ASE interface for Quantum Espresso"""

    implemented_properties = [
        'energy', 'free_energy', 'forces', 'stress', 'magmom', 'magmoms'
    ]

    def __init__(
            self,
            atoms=None,
            exedir='',  # espresso binary folder, if "./" just take the current environmental variable
            pw=350.0,
            dw=None,
            fw=None,
            nbands=-10,
            kpts=(1, 1, 1),
            kptshift=(0, 0, 0),
            fft_grid=None,
            mode=None,
            calculation='ase3',
            opt_algorithm=None,
            ion_dynamics='ase3',
            nstep=None,
            constr_tol=None,
            fmax=0.05,
            cell_dynamics=None,
            press=None,  # target pressure
            dpress=None,  # convergence limit towards target pressure
            cell_factor=None,
            cell_dofree=None,
            dontcalcforces=False,
            nosym=False,
            noinv=False,
            nosym_evc=False,
            no_t_rev=False,
            xc='PBE',
            beefensemble=False,
            printensemble=False,
            psppath=None,
            spinpol=False,
            noncollinear=False,
            spinorbit=False,
            outdir=None,
            txt=None,
            calcstress=False,
            smearing='fd',
            sigma=0.1,
            fix_magmom=False,
            isolated=None,
            U=None,
            J=None,
            U_alpha=None,
            U_projection_type='atomic',
            nqx1=None,
            nqx2=None,
            nqx3=None,
            exx_fraction=None,
            screening_parameter=None,
            exxdiv_treatment=None,
            ecutvcut=None,
            tot_charge=None,  # +1 means 1 e missing, -1 means 1 extra e
            charge=None,  # overrides tot_charge (ase 3.7+ compatibility)
            tot_magnetization=-1,  # -1 means unspecified, 'hund' means Hund's rule for each atom
            occupations='smearing',  # 'smearing', 'fixed', 'tetrahedra'
            dipole={'status': False},
            field={'status': False},
            output={
                'disk_io':
                'default',  # how often espresso writes wavefunctions to disk
                'avoidio': False,  # will overwrite disk_io parameter if True
                'removewf': True,
                'removesave': False,
                'wf_collect': False
            },
            convergence={
                'energy': 1e-6,
                'mixing': 0.7,
                'maxsteps': 100,
                'diag': 'david'
            },
            startingpot=None,
            startingwfc=None,
            ion_positions=None,
            parflags=None,
            onlycreatepwinp=None,  # specify filename to only create pw input
            single_calculator=True,  # if True, only one espresso job will be running
            procrange=None,  # let this espresso calculator run only on a subset of the requested cpus
            numcalcs=None,  # used / set by multiespresso class
            alwayscreatenewarrayforforces=True,
            verbose='low',
            # automatically generated list of parameters
            # some coincide with ase-style names
            iprint=None,
            tstress=None,
            tprnfor=None,
            dt=None,
            lkpoint_dir=None,
            max_seconds=None,
            etot_conv_thr=None,
            forc_conv_thr=None,
            tefield=None,
            dipfield=None,
            lelfield=None,
            nberrycyc=None,
            lorbm=None,
            lberry=None,
            gdir=None,
            nppstr=None,
            nbnd=None,
            ecutwfc=None,
            ecutrho=None,
            ecutfock=None,
            force_symmorphic=None,
            use_all_frac=None,
            one_atom_occupations=None,
            starting_spin_angle=None,
            degauss=None,
            nspin=None,
            ecfixed=None,
            qcutz=None,
            q2sigma=None,
            x_gamma_extrapolation=None,
            lda_plus_u=None,
            lda_plus_u_kind=None,
            edir=None,
            emaxpos=None,
            eopreg=None,
            eamp=None,
            clambda=None,
            report=None,
            lspinorb=None,
            esm_bc=None,
            esm_w=None,
            esm_efield=None,
            esm_nfit=None,
            london=None,
            london_s6=None,
            london_rcut=None,
            xdm=None,
            xdm_a1=None,
            xdm_a2=None,
            electron_maxstep=None,
            scf_must_converge=None,
            conv_thr=None,
            adaptive_thr=None,
            conv_thr_init=None,
            conv_thr_multi=None,
            mixing_beta=None,
            mixing_ndim=None,
            mixing_fixed_ns=None,
            ortho_para=None,
            diago_thr_init=None,
            diago_cg_maxiter=None,
            diago_david_ndim=None,
            diago_full_acc=None,
            efield=None,
            tqr=None,
            remove_rigid_rot=None,
            tempw=None,
            tolp=None,
            delta_t=None,
            nraise=None,
            refold_pos=None,
            upscale=None,
            bfgs_ndim=None,
            vdw_corr=None,
            ts_vdw_econv_thr=None,
            ts_vdw_isolated=None,
            lfcpopt=None,
            fcp_mu=None,
            esm_a=None,
            trust_radius_max=None,
            trust_radius_min=None,
            trust_radius_ini=None,
            w_1=None,
            w_2=None,
            wmass=None,
            press_conv_thr=None,
            results={},
            name='espresso',
            restart=None,
            ignore_bad_restart_file=False,
            label=None,
            command=None,
            nproc=None,
            # ENVIRON PART (credit Stefan Ringe)
            environ_keys=None,  # Environ keys given as dictionary, if given use_environ=True
            environ_extra_keys=None
    ):
        """Construct an ase-espresso calculator.

        Parameters (with defaults in parentheses):
        atoms (None)
           list of atoms object to be attached to calculator
           atoms.set_calculator can be used instead
        onlycreatepwinp (None)
           if not None but 'filename', create input file 'filename' for pw.x
           but do not run pw.x
           calc.initialize(atoms) will trigger 'filename' to be written
        pw (350.0)
           plane-wave cut-off in eV
        dw (10*pw)
           charge-density cut-off in eV
        fw (None)
           plane-wave cutoff for evaluation of EXX in eV
        nbands (-10)
           number of bands, if negative: -n extra bands
        kpts ( (1,1,1) )
           k-point grid sub-divisions, k-point grid density,
           explicit list of k-points, or simply 'gamma' for gamma-point only.
        kptshift ( (0,0,0) )
           shift of k-point grid
        fft_grid ( None )
           specify tuple of fft grid points (nr1,nr2,nr3) for q.e.
           useful for series of calculations with changing cell size
           (e.g. lattice constant optimization) uses q.e. default if not
           specified. [RK]
        mode (None)
           Deprecated, See calculation.
        calculation ( 'ase3' )
           relaxation mode:
           - 'ase3': dynamic communication between Quantum Espresso and python
           - 'relax', 'scf', 'nscf': corresponding Quantum Espresso standard modes
        opt_algorithm (None)
           Deprecated, See ion_dynamics.
        ion_dynamics ( 'ase3' )
           - 'ase3': ase updates coordinates during relaxation
           - 'relax' and other Quantum Espresso standard relaxation modes:
                     Quantum Espresso own algorithms for structural optimization
                     are used
           Obtaining Quantum Espresso with the ase3 relaxation extensions is
           highly recommended, since it allows for using ase's optimizers without
           loosing efficiency:
           svn co --username anonymous http://qeforge.qe-forge.org/svn/q-e/branches/espresso-dynpy-beef
        fmax (0.05)
           max force limit for Espresso-internal relaxation (eV/Angstrom)
        constr_tol (None)
           constraint tolerance for Espresso-internal relaxation
        cell_dynamics (None)
           algorithm (e.g. 'BFGS') to be used for Espresso-internal
           unit-cell optimization
        press (None)
           target pressure for such an optimization
        dpress (None)
           convergence limit towards target pressure
        cell_factor (None)
           should be >>1 if unit-cell volume is expected to shrink a lot during
           relaxation (would be more efficient to start with a better guess)
        cell_dofree (None)
           partially fix lattice vectors
        nosym (False)
        noinv (False)
        nosym_evc (False)
        no_t_rev (False)
           turn off corresp. symmetries
        xc ('PBE')
           xc-functional to be used
        beefensemble (False)
           calculate basis energies for ensemble error estimates based on
           the BEEF-vdW functional
        printensemble (False)
           let Espresso itself calculate 2000 ensemble energies
        psppath (None)
           Directory containing the pseudo-potentials or paw-setups to be used.
           The ase-espresso interface expects all pot. files to be of the type
           element.UPF (e.g. H.UPF).
           If None, the directory pointed to be ESP_PSP_PATH is used.
        spinpol (False)
           If True, calculation is spin-polarized
        noncollinear (False)
           Non-collinear magnetism.
        spinorbit (False)
           If True, spin-orbit coupling is considered.
           Make sure to provide j-dependent pseudo-potentials in psppath
           for those elements where spin-orbit coupling is important
        outdir (None)
           directory where Espresso's output is collected,
           default: qe<random>
        txt (None)
           If not None, direct Espresso's output to a different file than
           outdir/log
        calcstress (False)
           If True, calculate stress
        occupations ('smearing')
           Controls how Kohn-Sham states are occupied.
           Possible values: 'smearing', 'fixed' (molecule or insulator),
           or 'tetrahedra'.
        smearing ('fd')
           method for Fermi surface smearing
           - 'fd','Fermi-Dirac': Fermi-Dirac
           - 'mv','Marzari-Vanderbilt': Marzari-Vanderbilt cold smearing
           - 'gauss','gaussian': Gaussian smearing
           - 'mp','Methfessel-Paxton': Methfessel-Paxton
           For ase 3.7+ compatibility, smearing can also be a tuple where
           the first parameter is the method and the 2nd parameter
           is the smearing width which overrides sigma below
        sigma (0.1)
           smearing width in eV
        tot_charge (None)
           charge the unit cell,
           +1 means 1 e missing, -1 means 1 extra e
        charge (None)
           overrides tot_charge (ase 3.7+ compatibility)
        tot_magnetization (-1)
           Fix total magnetization,
           -1 means unspecified/free,
           'hund' means Hund's rule for each atom
        fix_magmom (False)
           If True, fix total magnetization to current value.
        isolated (None)
           invoke an 'assume_isolated' method for screening long-range interactions
           across 3D supercells, particularly electrostatics.
           Very useful for charged molecules and charged surfaces,
           but also improves convergence wrt. vacuum space for neutral molecules.
           - 'makov-payne', 'mp': only cubic systems.
           - 'dcc': don't use.
           - 'martyna-tuckerman', 'mt': method of choice for molecules, works for any supercell geometry.
           - 'esm': Effective Screening Medium Method for surfaces and interfaces.
        U (None)
           specify Hubbard U values (in eV)
           U can be list: specify U for each atom
           U can be a dictionary ( e.g. U={'Fe':3.5} )
           U values are assigned to angular momentum channels
           according to Espresso's hard-coded defaults
           (i.e. l=2 for transition metals, l=1 for oxygen, etc.)
        J (None)
           specify exchange J values (in eV)
           can be list or dictionary (see U parameter above)
        U_alpha
           U_alpha (in eV)
           can be list or dictionary (see U parameter above)
        U_projection_type ('atomic')
           type of projectors for calculating density matrices in DFT+U schemes
        nqx1, nqx2, nqx3 (all None)
           3D mesh for q=k1-k2 sampling of Fock operator. Can be smaller
           than number of k-points.
        exx_fraction (None)
           Default depends on hybrid functional chosen.
        screening_parameter (0.106)
           Screening parameter for HSE-like functionals.
        exxdiv_treatment (gygi-baldereschi)
           Method to treat Coulomb potential divergence for small q.
        ecutvcut (0)
           Cut-off for above.
        dipole ( {'status':False} )
           If 'status':True, turn on dipole correction; then by default, the
           dipole correction is applied along the z-direction, and the dipole is
           put in the center of the vacuum region (taking periodic boundary
           conditions into account).
           This can be overridden with:
           - 'edir':1, 2, or 3 for x-, y-, or z-direction
           - 'emaxpos':float percentage wrt. unit cell where dip. correction
             potential will be max.
           - 'eopreg':float percentage wrt. unit cell where potential decreases
           - 'eamp':0 (by default) if non-zero overcompensate dipole: i.e. apply
             a field
        output ( {'disk_io':'default',  # how often espresso writes wavefunctions to disk
                  'avoidio':False,  # will overwrite disk_io parameter if True
                  'removewf':True,
                  'removesave':False,
                  'wf_collect':False} )
           control how much io is used by espresso;
           'removewf':True means wave functions are deleted in scratch area before
           job is done and data is copied back to submission directory
           'removesave':True means whole .save directory is deleted in scratch area
        convergence ( {'energy':1e-6,
                       'mixing':0.7,
                       'maxsteps':100,
                       'diag':'david'} )
           Electronic convergence criteria and diag. and mixing algorithms.
           Additionally, a preconditioner for the mixing algoritms can be
           specified, e.g. 'mixing_mode':'local-TF' or 'mixing_mode':'TF'.
        startingpot (None)
           By default: 'atomic' (use superposition of atomic orbitals for
           initial guess)
           'file': construct potential from charge-density.dat
           Can be used with load_chg and save_chg methods.
        startingwfc (None)
           By default: 'atomic'.
           Other options: 'atomic+random' or 'random'.
           'file': reload wave functions from other calculations.
           See load_wf and save_wf methods.
        parflags (None)
           Parallelization flags for Quantum Espresso.
           E.g. parflags='-npool 2' will distribute k-points (and spin if
           spin-polarized) over two nodes.
        verbose ('low')
           Can be 'high' or 'low'
        """
        self.exedir = exedir
        self.outdir = outdir
        self.onlycreatepwinp = onlycreatepwinp
        self.pw = pw
        self.dw = dw
        self.fw = fw
        self.nbands = nbands
        if type(kpts) == float or type(kpts) == int:
            kpts = kptdensity2monkhorstpack(atoms, kpts)
        elif isinstance(kpts, str):
            assert kpts == 'gamma'
        else:
            assert len(kpts) == 3
        self.kpts = kpts
        self.kptshift = kptshift
        self.fft_grid = fft_grid  # RK
        if mode is not None and calculation is None:
            warnings.warn(
                "mode is deprecated, use calculation instead",
                DeprecationWarning)
            calculation = mode
        self.calculation = calculation
        if opt_algorithm is not None and ion_dynamics is None:
            warnings.warn(
                "opt_algorithm is deprecated, use ion_dynamics instead",
                DeprecationWarning)
        self.ion_dynamics = ion_dynamics
        self.nstep = nstep
        self.constr_tol = constr_tol
        self.fmax = fmax
        self.cell_dynamics = cell_dynamics
        self.press = press
        self.dpress = dpress
        self.cell_factor = cell_factor
        self.cell_dofree = cell_dofree
        self.dontcalcforces = dontcalcforces
        self.nosym = nosym
        self.noinv = noinv
        self.nosym_evc = nosym_evc
        self.no_t_rev = no_t_rev
        self.xc = xc
        self.beefensemble = beefensemble
        self.printensemble = printensemble
        if isinstance(smearing, str):
            self.smearing = str(smearing)
            self.sigma = sigma
        else:
            self.smearing = smearing[0]
            self.sigma = smearing[1]
        self.spinpol = spinpol
        self.noncollinear = noncollinear
        self.spinorbit = spinorbit
        self.fix_magmom = fix_magmom
        self.isolated = isolated
        if charge is None:
            self.tot_charge = tot_charge
        else:
            self.tot_charge = charge
        self.tot_magnetization = tot_magnetization
        self.occupations = occupations
        self.outdir = outdir
        self.calcstress = calcstress
        if psppath:
            self.psppath = psppath
        else:
            self.psppath = os.path.join(rootpath,'pp',self.xc)
        self.dipole = dipole
        self.field = field
        self.output = output
        self.convergence = convergence
        self.startingpot = startingpot
        self.startingwfc = startingwfc
        self.ion_positions = ion_positions
        self.verbose = verbose
        self.U = U
        self.J = J
        self.U_alpha = U_alpha
        self.U_projection_type = U_projection_type
        self.nqx1 = nqx1
        self.nqx2 = nqx2
        self.nqx3 = nqx3
        self.exx_fraction = exx_fraction
        self.screening_parameter = screening_parameter
        self.exxdiv_treatment = exxdiv_treatment
        self.ecutvcut = ecutvcut
        self.newforcearray = alwayscreatenewarrayforforces
        self.parflags = ''
        self.serflags = ''
        # ENVIRON IMPLICIT SOLVATION
        if environ_keys is not None:
            self.parflags = ' -environ '
            self.serflags = ' -environ '
            self.environ_keys = environ_keys
            self.use_environ = True
        else:
            self.parflags = ''
            self.serflags = ''
            self.use_environ = False
        self.environ_extra_keys=environ_extra_keys
        if parflags is not None:
            self.parflags += parflags
        self.single_calculator = single_calculator
        self.txt = txt
        self.writeversion = False
        self.atoms = None
        self.sigma_small = 1e-13
        self.started = False
        self.got_energy = False
        self.only_init = False

        # automatically generated list
        self.iprint = iprint
        self.tstress = tstress
        self.tprnfor = tprnfor
        self.dt = dt
        self.lkpoint_dir = lkpoint_dir
        self.max_seconds = max_seconds
        self.etot_conv_thr = etot_conv_thr
        self.forc_conv_thr = forc_conv_thr
        self.tefield = tefield
        self.dipfield = dipfield
        self.lelfield = lelfield
        self.nberrycyc = nberrycyc
        self.lorbm = lorbm
        self.lberry = lberry
        self.gdir = gdir
        self.nppstr = nppstr
        self.nbnd = nbnd
        self.ecutwfc = ecutwfc
        self.ecutrho = ecutrho
        self.ecutfock = ecutfock
        self.force_symmorphic = force_symmorphic
        self.use_all_frac = use_all_frac
        self.one_atom_occupations = one_atom_occupations
        self.starting_spin_angle = starting_spin_angle
        self.degauss = degauss
        self.nspin = nspin
        self.ecfixed = ecfixed
        self.qcutz = qcutz
        self.q2sigma = q2sigma
        self.x_gamma_extrapolation = x_gamma_extrapolation
        self.lda_plus_u = lda_plus_u
        self.lda_plus_u_kind = lda_plus_u_kind
        self.edir = edir
        self.emaxpos = emaxpos
        self.eopreg = eopreg
        self.eamp = eamp
        self.clambda = clambda
        self.report = report
        self.lspinorb = lspinorb
        self.esm_bc = esm_bc
        self.esm_w = esm_w
        self.esm_efield = esm_efield
        self.esm_nfit = esm_nfit
        self.london = london
        self.london_s6 = london_s6
        self.london_rcut = london_rcut
        self.xdm = xdm
        self.xdm_a1 = xdm_a1
        self.xdm_a2 = xdm_a2
        self.electron_maxstep = electron_maxstep
        self.scf_must_converge = scf_must_converge
        self.conv_thr = conv_thr
        self.adaptive_thr = adaptive_thr
        self.conv_thr_init = conv_thr_init
        self.conv_thr_multi = conv_thr_multi
        self.mixing_beta = mixing_beta
        self.mixing_ndim = mixing_ndim
        self.mixing_fixed_ns = mixing_fixed_ns
        self.ortho_para = ortho_para
        self.diago_thr_init = diago_thr_init
        self.diago_cg_maxiter = diago_cg_maxiter
        self.diago_david_ndim = diago_david_ndim
        self.diago_full_acc = diago_full_acc
        self.efield = efield
        self.tqr = tqr
        self.remove_rigid_rot = remove_rigid_rot
        self.tempw = tempw
        self.tolp = tolp
        self.delta_t = delta_t
        self.nraise = nraise
        self.refold_pos = refold_pos
        self.upscale = upscale
        self.bfgs_ndim = bfgs_ndim
        self.vdw_corr = vdw_corr
        self.ts_vdw_econv_thr = ts_vdw_econv_thr
        self.ts_vdw_isolated = ts_vdw_isolated
        self.lfcpopt = lfcpopt
        self.fcp_mu = fcp_mu
        self.esm_a = esm_a
        self.trust_radius_max = trust_radius_max
        self.trust_radius_min = trust_radius_min
        self.trust_radius_ini = trust_radius_ini
        self.w_1 = w_1
        self.w_2 = w_2
        self.wmass = wmass
        self.press_conv_thr = press_conv_thr
        self.results = results
        self.name = name

        # set up the site object
        total_cpus = multiprocessing.cpu_count()
        if not nproc or nproc > total_cpus:
            nproc = total_cpus
        self.site = espsite.Config(nproc)

        # give original espresso style input names
        # preference over ase / dacapo - style names
        if ecutwfc is not None:
            self.pw = ecutwfc
        if ecutrho is not None:
            self.dw = ecutwfc
        if nbnd is not None:
            self.nbands = nbnd

        # Variables that cannot be set by inputs
        self.nvalence = None
        self.nel = None
        self.fermi_input = False

        self.parameters = {}

        # Auto create variables from input
        self.input_update()

        # Initialize lists of cpu subsets if needed
        if procrange is None:
            self.proclist = False
        else:
            self.proclist = True
            procs = self.site.procs + []
            procs.sort()
            nprocs = len(procs)
            self.myncpus = nprocs / numcalcs
            i1 = self.myncpus * procrange
            self.mycpus = self.localtmp + '/myprocs%04d.txt' % procrange
            f = open(self.mycpus, 'w')
            for i in range(i1, i1 + self.myncpus):
                print(procs[i], file=f)
            f.close()

        if atoms is not None:
            atoms.set_calculator(self)

        if hasattr(self.site, 'mpi_not_setup') and self.onlycreatepwinp is None:
            print(
                '*** Without cluster-adjusted espsite.py, ase-espresso can only be used\n'
                '*** to create input files for pw.x via the option onlycreatepwinp.\n'
                '*** Otherwise, ase-espresso requires a site-specific espsite.py\n'
                '*** in PYTHONPATH.\n'
                '*** You may use the espsite.py.example.* in the git checkout as templates.'
            )
            raise ImportError

    def input_update(self):
        """Run initialization functions, such that this can be called
        if variables in espresso are changes using set or directly.
        """
        self.create_outdir()  # Create the tmp output folder

        # sdir is the directory the script is run or submitted from
        self.sdir = subdirs.getsubmitorcurrentdir(self.site)

        if self.dw is None:
            self.dw = 10. * self.pw
        else:
            assert self.dw >= self.pw

        if self.psppath is None:
            try:
                self.psppath = os.environ['ESP_PSP_PATH']
            except BaseException:
                print(
                    'Unable to find pseudopotential path.  Consider '
                    'setting ESP_PSP_PATH environment variable'
                )
                raise
        if self.dipole is None:
            self.dipole = {'status': False}
        if self.field is None:
            self.field = {'status': False}

        if self.convergence is None:
            self.conv_thr = 1e-6 / Rydberg
        else:
            if 'energy' in self.convergence:
                self.conv_thr = self.convergence['energy'] / Rydberg
            else:
                self.conv_thr = 1e-6 / Rydberg

        if self.beefensemble:
            if self.xc.upper().find('BEEF') < 0:
                raise KeyError(
                    "ensemble-energies only work with xc=BEEF or variants of it!"
                )

        self.started = False
        self.got_energy = False

    def create_outdir(self):
        if self.onlycreatepwinp is None:
            self.localtmp = subdirs.mklocaltmp(self.outdir, self.site)
            if not self.txt:
                self.log = self.localtmp + '/log'
            elif self.txt[0] != '/':
                self.log = self.sdir+'/'+self.txt
            else:
                self.log = self.txt
            self.scratch = subdirs.mkscratch(self.localtmp, self.site)
            if self.output is not None:
                if 'removewf' in self.output:
                    removewf = self.output['removewf']
                else:
                    removewf = True
                if 'removesave' in self.output:
                    removesave = self.output['removesave']
                else:
                    removesave = False
            else:
                removewf = True
                removesave = False
            atexit.register(subdirs.cleanup, self.localtmp, self.scratch,
                            removewf, removesave, self, self.site)
            self.cancalc = True
        else:
            self.pwinp = self.onlycreatepwinp
            self.localtmp = ''
            self.cancalc = False

    def set(self, **kwargs):
        """Define settings for the Quantum Espresso calculator object
        after it has been initialized. This is done in the following way:

        >> calc = espresso(...)
        >> atoms = set.calculator(calc)
        >> calc.set(xc='BEEF')

        NB: No input validation is made
        """
        for key, value in list(kwargs.items()):
            if key == 'outdir':
                self.create_outdir()
                self.outdir = value
            if key == 'startingpot':
                self.startingpot = value
            if key == 'startingwfc':
                self.startingwfc = value
            if key == 'ion_positions':
                self.ion_positions = value
            if key == 'U_alpha':
                self.U_alpha = value
            if key == 'U':
                self.U = value
            if key == 'U_projection_type':
                self.U_projection_type = value
            if key == 'xc':
                self.xc = value
            if key == 'pw':
                self.pw = value
            if key == 'dw':
                self.dw = value
            if key == 'output':
                self.output = value
            if key == 'convergence':
                self.convergence = value
            if key == 'kpts':
                self.kpts = value
            if key == 'kshift':
                self.kshift = value
            if key == 'fft_grid':  # RK
                self.fft_grid = value
        self.input_update()
        self.recalculate = True
        self.results = {}

    def __del__(self):
        try:
            self.stop()
        except BaseException:
            pass

    def atoms2species(self):
        """Define several properties of the quantum espresso species
        from the ase atoms object. Takes into account that different
        spins (or different U etc.) on same kind of chemical elements
        are considered different species in quantum espresso.
        """
        symbols = self.atoms.get_chemical_symbols()
        masses = self.atoms.get_masses()
        magmoms = list(self.atoms.get_initial_magnetic_moments())
        if len(magmoms) < len(symbols):
            magmoms += list(np.zeros(len(symbols) - len(magmoms), np.float))
        pos = self.atoms.get_scaled_positions()

        if self.U is not None:
            if type(self.U) == dict:
                Ulist = np.zeros(len(symbols), np.float)
                for i, s in enumerate(symbols):
                    if s in self.U:
                        Ulist[i] = self.U[s]
            else:
                Ulist = list(self.U)
                if len(Ulist) < len(symbols):
                    Ulist += list(
                        np.zeros(len(symbols) - len(Ulist), np.float))
        else:
            Ulist = np.zeros(len(symbols), np.float)

        if self.J is not None:
            if type(self.J) == dict:
                Jlist = np.zeros(len(symbols), np.float)
                for i, s in enumerate(symbols):
                    if s in self.J:
                        Jlist[i] = self.J[s]
            else:
                Jlist = list(self.J)
                if len(Jlist) < len(symbols):
                    Jlist += list(
                        np.zeros(len(symbols) - len(Jlist), np.float))
        else:
            Jlist = np.zeros(len(symbols), np.float)

        if self.U_alpha is not None:
            if type(self.U_alpha) == dict:
                U_alphalist = np.zeros(len(symbols), np.float)
                for i, s in enumerate(symbols):
                    if s in self.U_alpha:
                        U_alphalist[i] = self.U_alpha[s]
            else:
                U_alphalist = list(self.U_alpha)
                if len(U_alphalist) < len(symbols):
                    U_alphalist += list(
                        np.zeros(len(symbols) - len(U_alphalist), np.float))
        else:
            U_alphalist = np.zeros(len(symbols), np.float)

        self.species = []
        self.specprops = []
        dic = {}
        symcounter = {}
        for s in symbols:
            symcounter[s] = 0
        for i in range(len(symbols)):
            key = symbols[i] + '_m%.14eU%.14eJ%.14eUa%.14e' % (
                magmoms[i], Ulist[i], Jlist[i], U_alphalist[i])
            if key in dic:
                self.specprops.append((dic[key][1], pos[i]))
            else:
                symcounter[symbols[i]] += 1
                spec = symbols[i] + str(symcounter[symbols[i]])
                dic[key] = [i, spec]
                self.species.append(spec)
                self.specprops.append((spec, pos[i]))

        self.nspecies = len(self.species)
        self.specdict = {}
        for i, s in list(dic.values()):
            if np.isnan(masses[i]):
                mi = 0.0
            else:
                mi = masses[i]
            self.specdict[s] = utils.SpecObject(
                s=s.strip('0123456789'),  # chemical symbol w/o index
                mass=mi,
                magmom=magmoms[i],
                U=Ulist[i],
                J=Jlist[i],
                U_alpha=U_alphalist[i])

    def get_nvalence(self):
        nel = {}
        for x in self.species:
            el = self.specdict[x].s
            # get number of valence electrons from pseudopotential or paw setup
            p = Popen(
                'egrep -i \'z\ valence|z_valence\' ' + self.psppath +
                '/' + el + '.UPF | tr \'"\' \' \'',
                shell=True, stdout=PIPE).stdout
            out = p.readline().decode('utf-8')
            for y in out.split():
                if y[0].isdigit() or y[0] == '.':
                    nel[el] = int(round(float(y)))
                    break
            p.close()
        nvalence = np.zeros(len(self.specprops), np.int)
        for i, x in enumerate(self.specprops):
            nvalence[i] = nel[self.specdict[x[0]].s]
        return nvalence, nel

    def writeenvinputfile(self, filename='environ.in'):
        """Write Environ input file"""
        if self.cancalc:
            fname = self.localtmp+'/'+filename
            #f = open(self.localtmp+'/pw.inp', 'w')
        else:
            fname = self.pwinp.split('/')[:-1]+'/'+filename
        f = open(fname,'w')
        f.write('&ENVIRON\n')
        f.write('  !\n')
        for key in self.environ_keys:
            value=self.environ_keys[key]
            if type(value)==bool:
                if value:
                    value='.true.'
                else:
                    value='.false.'
                f.write('  {} = {}\n'.format(key,value))
            elif type(value)==str:
                f.write('  {} = \'{}\'\n'.format(key, value))
            elif 'e' in str(value):
                value_str=str(value).replace('e','D')
                f.write('  {} = {}\n'.format(key, value_str))
            else:
                f.write('  {} = {}\n'.format(key, self.environ_keys[key]))
        f.write('  !\n')
        f.write('/\n')
        if self.environ_extra_keys is not None:
            for key in self.environ_extra_keys:
                #f.write('{} {}\n'.format(key,unit))
                if key in ['EXTERNAL_CHARGES','DIELECTRIC_REGIONS']:
                    #CARDs
                    if 'unit' not in self.environ_extra_keys[key]:
                        unit='bohr'
                    else:
                        unit=self.environ_extra_keys[key]['unit']
                    f.write('{} {}\n'.format(key,unit))
                    for vals in self.environ_extra_keys[key]['settings']:
                        f.write('  {}\n'.format(' '.join([str(vv) for vv in vals])))
                else:
                    #NAMELISTs
                    f.write('&{}\n'.format(key))
                    f.write('  !\n')
                    for key2 in self.environ_extra_keys[key]:
                        value=self.environ_extra_keys[key][key2]
                        if type(value)==bool:
                            if value:
                                value='.true.'
                            else:
                                value='.false.'
                            f.write('  {} = {}\n'.format(key2,value))
                        elif type(value)==str:
                            f.write('  {} = \'{}\'\n'.format(key2,value))
                        elif 'e' in str(value):
                            value_str=str(value).replace('e','D')
                            f.write('  {} = {}\n'.format(key2, value_str))
                        else:
                            f.write('  {} = {}\n'.format(key2,value))
                    f.write('  !\n')
                    f.write('/\n')
        f.close()

    def writeinputfile(self,
                       filename='pw.inp',
                       mode=None,
                       overridekpts=None,
                       overridekptshift=None,
                       overridenbands=None,
                       suppressforcecalc=False,
                       usetetrahedra=False):
        if self.atoms is None:
            raise ValueError('no atoms defined')
        if self.cancalc:
            fname = self.localtmp + '/' + filename
        else:
            fname = self.pwinp
        f = open(fname, 'w')

        # &CONTROL ###
        if mode is None:
            if self.calculation == 'ase3':
                print(
                    '&CONTROL\n  calculation=\'relax\',\n  prefix=\'calc\',',
                    file=f)
            elif self.calculation == 'hund':
                print(
                    '&CONTROL\n  calculation=\'scf\',\n  prefix=\'calc\',',
                    file=f)
            else:
                print(
                    '&CONTROL\n  calculation=\'' + self.calculation +
                    '\',\n  prefix=\'calc\',',
                    file=f)
            ionssec = self.calculation not in ('scf', 'nscf', 'bands', 'hund')
        else:
            print(
                '&CONTROL\n  calculation=\'' + mode +
                '\',\n  prefix=\'calc\',',
                file=f)
            ionssec = mode not in ('scf', 'nscf', 'bands', 'hund')

        if self.nstep is not None:
            print('  nstep=' + str(self.nstep) + ',', file=f)

        if self.verbose != 'low':
            print('  verbosity=\'' + self.verbose + '\',', file=f)

        print('  pseudo_dir=\'' + self.psppath + '\',', file=f)
        print('  outdir=\'.\',', file=f)
        efield = (self.field['status'])
        dipfield = (self.dipole['status'])
        if efield or dipfield:
            print('  tefield=.true.,', file=f)
            if dipfield:
                print('  dipfield=.true.,', file=f)
        if not self.dontcalcforces and not suppressforcecalc:
            print('  tprnfor=.true.,', file=f)
            if self.calcstress:
                print('  tstress=.true.,', file=f)
            if self.output is not None:
                if 'avoidio' in self.output:
                    if self.output['avoidio']:
                        self.output['disk_io'] = 'none'
                if 'disk_io' in self.output:
                    if self.output['disk_io'] in ['high', 'low', 'none']:
                        print(
                            '  disk_io=\'' + self.output['disk_io'] + '\',',
                            file=f)

                if 'wf_collect' in self.output:
                    if self.output['wf_collect']:
                        print('  wf_collect=.true.,', file=f)
        if self.ion_dynamics != 'ase3' or not self.cancalc:
            # We basically ignore convergence of total energy differences
            # between ionic steps and only consider fmax as in ase
            print('  etot_conv_thr=1d0,', file=f)
            print(
                '  forc_conv_thr=' + utils.num2str(self.fmax /
                                                   (Rydberg / Bohr)) + ',',
                file=f)

        # turn on fifo communication if espsite.py is set up that way
        if hasattr(self.site, 'fifo'):
            if self.site.fifo:
                print('  ase_fifo=.true.,', file=f)

        # automatically generated parameters
        if self.iprint is not None:
            print('  iprint=' + str(self.iprint) + ',', file=f)
        if self.tstress is not None:
            print('  tstress=' + utils.bool2str(self.tstress) + ',', file=f)
        if self.tprnfor is not None:
            print('  tprnfor=' + utils.bool2str(self.tprnfor) + ',', file=f)
        if self.dt is not None:
            print('  dt=' + utils.num2str(self.dt) + ',', file=f)
        if self.lkpoint_dir is not None:
            print(
                '  lkpoint_dir=' + utils.bool2str(self.lkpoint_dir) + ',',
                file=f)
        if self.max_seconds is not None:
            print(
                '  max_seconds=' + utils.num2str(self.max_seconds) + ',',
                file=f)
        if self.etot_conv_thr is not None:
            print(
                '  etot_conv_thr=' + utils.num2str(self.etot_conv_thr) + ',',
                file=f)
        if self.forc_conv_thr is not None:
            print(
                '  forc_conv_thr=' + utils.num2str(self.forc_conv_thr) + ',',
                file=f)
        if self.tefield is not None:
            print('  tefield=' + utils.bool2str(self.tefield) + ',', file=f)
        if self.dipfield is not None:
            print('  dipfield=' + utils.bool2str(self.dipfield) + ',', file=f)
        if self.lelfield is not None:
            print('  lelfield=' + utils.bool2str(self.lelfield) + ',', file=f)
        if self.nberrycyc is not None:
            print('  nberrycyc=' + str(self.nberrycyc) + ',', file=f)
        if self.lorbm is not None:
            print('  lorbm=' + utils.bool2str(self.lorbm) + ',', file=f)
        if self.lberry is not None:
            print('  lberry=' + utils.bool2str(self.lberry) + ',', file=f)
        if self.gdir is not None:
            print('  gdir=' + str(self.gdir) + ',', file=f)
        if self.nppstr is not None:
            print('  nppstr=' + str(self.nppstr) + ',', file=f)
        if self.lfcpopt is not None:
            print('  lfcpopt=' + utils.bool2str(self.lfcpopt) + ',', file=f)

        ### &SYSTEM ###
        print('/\n&SYSTEM\n  ibrav=0,', file=f)
        print('  nat=' + str(self.natoms) + ',', file=f)
        self.atoms2species()  # self.convertmag2species()
        print(
            '  ntyp=' + str(self.nspecies) + ',',
            file=f)  # str(len(self.msym))+','
        if self.tot_charge is not None:
            print(
                '  tot_charge=' + utils.num2str(self.tot_charge) + ',', file=f)
        if self.calculation != 'hund':
            inimagscale = 1.0
        else:
            inimagscale = 0.9
        if self.fix_magmom:
            assert self.spinpol
            self.totmag = self.summed_magmoms
            print(
                '  tot_magnetization=' +
                utils.num2str(self.totmag * inimagscale) + ',',
                file=f)
        elif self.tot_magnetization != -1:
            if self.tot_magnetization != 'hund':
                self.totmag = self.tot_magnetization
            else:
                self.totmag = sum(
                    [hundmag(x) for x in self.atoms.get_chemical_symbols()])
            print(
                '  tot_magnetization=' +
                utils.num2str(self.totmag * inimagscale) + ',',
                file=f)
        print('  ecutwfc=' + utils.num2str(self.pw / Rydberg) + ',', file=f)
        print('  ecutrho=' + utils.num2str(self.dw / Rydberg) + ',', file=f)
        if self.fw is not None:
            print(
                '  ecutfock=' + utils.num2str(self.fw / Rydberg) + ',', file=f)
        # temporarily (and optionally) change number of bands for nscf calc.
        if overridenbands is not None:
            if self.nbands is None:
                nbandssave = None
            else:
                nbandssave = self.nbands
            self.nbands = overridenbands
        if self.nbands is not None:
            # set number of bands
            if self.nbands > 0:
                self.nbnd = int(self.nbands)
            else:
                # if self.nbands is negative create -self.nbands extra bands
                if self.nvalence is None:
                    self.nvalence, self.nel = self.get_nvalence()
                if self.noncollinear:
                    self.nbnd = int(np.sum(self.nvalence) - self.nbands * 2.)
                else:
                    self.nbnd = int(np.sum(self.nvalence) / 2. - self.nbands)
            print('  nbnd=' + str(self.nbnd) + ',', file=f)
        if overridenbands is not None:
            self.nbands = nbandssave
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
                print(
                    '  starting_magnetization(%d)=%s,' %
                    (spcount, utils.num2str(float(mag))),
                    file=f)
                spcount += 1
        elif self.noncollinear:
            print('  noncolin=.true.,', file=f)
            if self.spinorbit:
                print('  lspinorb=.true.', file=f)
            spcount = 1
            if self.nel is None:
                self.nvalence, self.nel = self.get_nvalence()
            # FOLLOW SAME ORDERING ROUTINE AS FOR PSP
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
        edir = 3
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
        if (self.U is not None or
            self.J is not None or
            self.U_alpha is not None):
            print('  lda_plus_u=.true.,', file=f)
            if self.J is not None:
                print('  lda_plus_u_kind=1,', file=f)
            else:
                print('  lda_plus_u_kind=0,', file=f)
            print(
                '  U_projection_type=\"%s\",' % (self.U_projection_type),
                file=f)
            if self.U is not None:
                for i, s in enumerate(self.species):
                    spec = self.specdict[s]
                    el = spec.s
                    Ui = spec.U
                    print(
                        '  Hubbard_U(' + str(i + 1) + ')=' +
                        utils.num2str(Ui) + ',',
                        file=f)
            if self.J is not None:
                for i, s in enumerate(self.species):
                    spec = self.specdict[s]
                    el = spec.s
                    Ji = spec.J
                    print(
                        '  Hubbard_J(1,' + str(i + 1) + ')=' +
                        utils.num2str(Ji) + ',',
                        file=f)
            if self.U_alpha is not None:
                for i, s in enumerate(self.species):
                    spec = self.specdict[s]
                    el = spec.s
                    U_alphai = spec.U_alpha
                    print(
                        '  Hubbard_alpha(' + str(i + 1) + ')=' +
                        utils.num2str(U_alphai) + ',',
                        file=f)

        if self.nqx1 is not None:
            print('  nqx1=%d,' % self.nqx1, file=f)
        if self.nqx2 is not None:
            print('  nqx2=%d,' % self.nqx2, file=f)
        if self.nqx3 is not None:
            print('  nqx3=%d,' % self.nqx3, file=f)

        if self.exx_fraction is not None:
            print(
                '  exx_fraction=' + utils.num2str(self.exx_fraction) + ',',
                file=f)
        if self.screening_parameter is not None:
            print(
                '  screening_parameter=' +
                utils.num2str(self.screening_parameter) + ',',
                file=f)
        if self.exxdiv_treatment is not None:
            print(
                '  exxdiv_treatment=\'' + self.exxdiv_treatment + '\',',
                file=f)
        if self.ecutvcut is not None:
            print('  ecutvcut=' + utils.num2str(self.ecutvcut) + ',', file=f)

        if self.nosym:
            print('  nosym=.true.,', file=f)
        if self.noinv:
            print('  noinv=.true.,', file=f)
        if self.nosym_evc:
            print('  nosym_evc=.true.,', file=f)
        if self.no_t_rev:
            print('  no_t_rev=.true.,', file=f)
        if self.fft_grid is not None:  # RK
            print('  nr1=%d,' % self.fft_grid[0], file=f)
            print('  nr2=%d,' % self.fft_grid[1], file=f)
            print('  nr3=%d,' % self.fft_grid[2], file=f)

        # automatically generated parameters
        if self.ecutfock is not None:
            print('  ecutfock=' + utils.num2str(self.ecutfock) + ',', file=f)
        if self.force_symmorphic is not None:
            print(
                '  force_symmorphic=' + utils.bool2str(self.force_symmorphic) +
                ',',
                file=f)
        if self.use_all_frac is not None:
            print(
                '  use_all_frac=' + utils.bool2str(self.use_all_frac) + ',',
                file=f)
        if self.one_atom_occupations is not None:
            print(
                '  one_atom_occupations=' +
                utils.bool2str(self.one_atom_occupations) + ',',
                file=f)
        if self.starting_spin_angle is not None:
            print(
                '  starting_spin_angle=' +
                utils.bool2str(self.starting_spin_angle) + ',',
                file=f)
        if self.degauss is not None:
            print('  degauss=' + utils.num2str(self.degauss) + ',', file=f)
        if self.nspin is not None:
            print('  nspin=' + str(self.nspin) + ',', file=f)
        if self.ecfixed is not None:
            print('  ecfixed=' + utils.num2str(self.ecfixed) + ',', file=f)
        if self.qcutz is not None:
            print('  qcutz=' + utils.num2str(self.qcutz) + ',', file=f)
        if self.q2sigma is not None:
            print('  q2sigma=' + utils.num2str(self.q2sigma) + ',', file=f)
        if self.x_gamma_extrapolation is not None:
            print(
                '  x_gamma_extrapolation=' +
                utils.bool2str(self.x_gamma_extrapolation) + ',',
                file=f)
        if self.lda_plus_u is not None:
            print(
                '  lda_plus_u=' + utils.bool2str(self.lda_plus_u) + ',',
                file=f)
        if self.lda_plus_u_kind is not None:
            print(
                '  lda_plus_u_kind=' + str(self.lda_plus_u_kind) + ',', file=f)
        if self.edir is not None:
            print('  edir=' + str(self.edir) + ',', file=f)
        if self.emaxpos is not None:
            print('  emaxpos=' + utils.num2str(self.emaxpos) + ',', file=f)
        if self.eopreg is not None:
            print('  eopreg=' + utils.num2str(self.eopreg) + ',', file=f)
        if self.eamp is not None:
            print('  eamp=' + utils.num2str(self.eamp) + ',', file=f)
        if self.clambda is not None:
            print('  lambda=' + utils.num2str(self.clambda) + ',', file=f)
        if self.report is not None:
            print('  report=' + str(self.report) + ',', file=f)
        if self.lspinorb is not None:
            print('  lspinorb=' + utils.bool2str(self.lspinorb) + ',', file=f)
        if self.esm_bc is not None:
            print('  esm_bc=\'' + self.esm_bc + '\',', file=f)
        if self.esm_w is not None:
            print('  esm_w=' + utils.num2str(self.esm_w) + ',', file=f)
        if self.esm_efield is not None:
            print(
                '  esm_efield=' + utils.num2str(self.esm_efield) + ',', file=f)
        if self.esm_nfit is not None:
            print('  esm_nfit=' + str(self.esm_nfit) + ',', file=f)
        if self.london is not None:
            print('  london=' + utils.bool2str(self.london) + ',', file=f)
        if self.london_s6 is not None:
            print('  london_s6=' + utils.num2str(self.london_s6) + ',', file=f)
        if self.london_rcut is not None:
            print(
                '  london_rcut=' + utils.num2str(self.london_rcut) + ',',
                file=f)
        if self.xdm is not None:
            print('  xdm=' + utils.bool2str(self.xdm) + ',', file=f)
        if self.xdm_a1 is not None:
            print('  xdm_a1=' + utils.num2str(self.xdm_a1) + ',', file=f)
        if self.xdm_a2 is not None:
            print('  xdm_a2=' + utils.num2str(self.xdm_a2) + ',', file=f)
        if self.vdw_corr is not None:
            print('  vdw_corr=\'' + self.vdw_corr + '\',', file=f)
        if self.ts_vdw_econv_thr is not None:
            print(
                '  ts_vdw_econv_thr=' + utils.num2str(self.ts_vdw_econv_thr) +
                ',',
                file=f)
        if self.ts_vdw_isolated is not None:
            print(
                '  ts_vdw_isolated=' + utils.bool2str(self.tsw_vdw_isolated) +
                ',',
                file=f)
        if self.fcp_mu is not None:
            print('  fcp_mu=' + utils.num2str(self.fcp_mu) + ',', file=f)
        if self.esm_a is not None:
            print('  esm_a=' + utils.num2str(self.esm_a) + ',', file=f)

        # &ELECTRONS ###
        print('/\n&ELECTRONS', file=f)
        try:
            diag = self.convergence['diag']
            print('  diagonalization=\'' + diag + '\',', file=f)
        except BaseException:
            pass

        #if self.calculation != 'hund':
        #    print('  conv_thr=' + utils.num2str(self.conv_thr) + ',', file=f)
        #else:
        #    print(
        #        '  conv_thr=' + utils.num2str(self.conv_thr * 500.) + ',',
        #        file=f)
        for x in list(self.convergence.keys()):
            if x == 'mixing':
                print(
                    '  mixing_beta=' + utils.num2str(self.convergence[x]) +
                    ',',
                    file=f)
            elif x == 'maxsteps':
                print(
                    '  electron_maxstep=' + str(self.convergence[x]) + ',',
                    file=f)
            elif x == 'nmix':
                print(
                    '  mixing_ndim=' + str(self.convergence[x]) + ',', file=f)
            elif x == 'mixing_mode':
                print('  mixing_mode=\'' + self.convergence[x] + '\',', file=f)
            elif x == 'diago_cg_maxiter':
                print(
                    '  diago_cg_maxiter=' + str(self.convergence[x]) + ',',
                    file=f)
        if self.startingpot is not None and self.calculation != 'hund':
            print('  startingpot=\'' + self.startingpot + '\',', file=f)
        if self.startingwfc is not None and self.calculation != 'hund':
            print('  startingwfc=\'' + self.startingwfc + '\',', file=f)

        # automatically generated parameters
        if self.electron_maxstep is not None:
            print(
                '  electron_maxstep=' + str(self.electron_maxstep) + ',',
                file=f)
        if self.scf_must_converge is not None:
            print(
                '  scf_must_converge=' +
                utils.bool2str(self.scf_must_converge) + ',',
                file=f)
        if self.conv_thr is not None:
            print('  conv_thr=' + utils.num2str(self.conv_thr) + ',', file=f)
        if self.adaptive_thr is not None:
            print(
                '  adaptive_thr=' + utils.bool2str(self.adaptive_thr) + ',',
                file=f)
        if self.conv_thr_init is not None:
            print(
                '  conv_thr_init=' + utils.num2str(self.conv_thr_init) + ',',
                file=f)
        if self.conv_thr_multi is not None:
            print(
                '  conv_thr_multi=' + utils.num2str(self.conv_thr_multi) + ',',
                file=f)
        if self.mixing_beta is not None:
            print(
                '  mixing_beta=' + utils.num2str(self.mixing_beta) + ',',
                file=f)
        if self.mixing_ndim is not None:
            print('  mixing_ndim=' + str(self.mixing_ndim) + ',', file=f)
        if self.mixing_fixed_ns is not None:
            print(
                '  mixing_fixed_ns=' + str(self.mixing_fixed_ns) + ',', file=f)
        if self.ortho_para is not None:
            print('  ortho_para=' + str(self.ortho_para) + ',', file=f)
        if self.diago_thr_init is not None:
            print(
                '  diago_thr_init=' + utils.num2str(self.diago_thr_init) + ',',
                file=f)
        if self.diago_cg_maxiter is not None:
            print(
                '  diago_cg_maxiter=' + str(self.diago_cg_maxiter) + ',',
                file=f)
        if self.diago_david_ndim is not None:
            print(
                '  diago_david_ndim=' + str(self.diago_david_ndim) + ',',
                file=f)
        if self.diago_full_acc is not None:
            print(
                '  diago_full_acc=' + utils.bool2str(self.diago_full_acc) +
                ',',
                file=f)
        if self.efield is not None:
            print('  efield=' + utils.num2str(self.efield) + ',', file=f)
        if self.tqr is not None:
            print('  tqr=' + utils.bool2str(self.tqr) + ',', file=f)

        # &IONS ###
        if self.ion_dynamics == 'ase3' or not ionssec:
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
            elif self.cancalc:
                print(
                    '/\n&IONS\n  ion_dynamics=\'' + self.ion_dynamics + '\',',
                    file=f)
            else:
                print('/\n&IONS\n  ion_dynamics=\'bfgs\',', file=f)
            if self.ion_positions is not None:
                print(
                    '  ion_positions=\'' + self.ion_positions + '\',', file=f)
        elif self.ion_positions is not None:
            print(
                '/\n&IONS\n  ion_positions=\'' + self.ion_positions + '\',',
                file=f)

        # automatically generated parameters
        if self.remove_rigid_rot is not None:
            print(
                '  remove_rigid_rot=' + utils.bool2str(self.remove_rigid_rot) +
                ',',
                file=f)
        if self.tempw is not None:
            print('  tempw=' + utils.num2str(self.tempw) + ',', file=f)
        if self.tolp is not None:
            print('  tolp=' + utils.num2str(self.tolp) + ',', file=f)
        if self.delta_t is not None:
            print('  delta_t=' + utils.num2str(self.delta_t) + ',', file=f)
        if self.nraise is not None:
            print('  nraise=' + str(self.nraise) + ',', file=f)
        if self.refold_pos is not None:
            print(
                '  refold_pos=' + utils.bool2str(self.refold_pos) + ',',
                file=f)
        if self.upscale is not None:
            print('  upscale=' + utils.num2str(self.upscale) + ',', file=f)
        if self.bfgs_ndim is not None:
            print('  bfgs_ndim=' + str(self.bfgs_ndim) + ',', file=f)
        if self.trust_radius_max is not None:
            print(
                '  trust_radius_max=' + utils.num2str(self.trust_radius_max) +
                ',',
                file=f)
        if self.trust_radius_min is not None:
            print(
                '  trust_radius_min=' + utils.num2str(self.trust_radius_min) +
                ',',
                file=f)
        if self.trust_radius_ini is not None:
            print(
                '  trust_radius_ini=' + utils.num2str(self.trust_radius_ini) +
                ',',
                file=f)
        if self.w_1 is not None:
            print('  w_1=' + utils.num2str(self.w_1) + ',', file=f)
        if self.w_2 is not None:
            print('  w_2=' + utils.num2str(self.w_2) + ',', file=f)

        # &CELL ###
        if self.cell_dynamics is not None:
            print(
                '/\n&CELL\n  cell_dynamics=\'' + self.cell_dynamics + '\',',
                file=f)
            if self.press is not None:
                print('  press=' + utils.num2str(self.press) + ',', file=f)
            if self.dpress is not None:
                print(
                    '  press_conv_thr=' + utils.num2str(self.dpress) + ',',
                    file=f)
            if self.cell_factor is not None:
                print(
                    '  cell_factor=' + utils.num2str(self.cell_factor) + ',',
                    file=f)
            if self.cell_dofree is not None:
                print('  cell_dofree=\'' + self.cell_dofree + '\',', file=f)

        # automatically generated parameters
        if self.wmass is not None:
            print('  wmass=' + utils.num2str(self.wmass) + ',', file=f)
        if self.press_conv_thr is not None:
            print(
                '  press_conv_thr=' + utils.num2str(self.press_conv_thr) + ',',
                file=f)

        # CELL_PARAMETERS
        print('/\nCELL_PARAMETERS {angstrom}', file=f)
        for i in range(3):
            print(
                '%21.15fd0 %21.15fd0 %21.15fd0' % tuple(self.atoms.cell[i]),
                file=f)

        print('ATOMIC_SPECIES', file=f)
        for species in self.species:  # PSP ORDERING FOLLOWS SPECIESINDEX
            spec = self.specdict[species]
            print(species, utils.num2str(spec.mass), spec.s + '.UPF', file=f)

        print('ATOMIC_POSITIONS {crystal}', file=f)
        if len(simpleconstr) == 0:
            for species, pos in self.specprops:
                print(
                    '%-4s %21.15fd0 %21.15fd0 %21.15fd0' % (species, pos[0],
                                                            pos[1], pos[2]),
                    file=f)
        else:
            for i, (species, pos) in enumerate(self.specprops):
                print(
                    '%-4s %21.15fd0 %21.15fd0 %21.15fd0   %d  %d  %d' %
                    (species, pos[0], pos[1], pos[2], simpleconstr[i][0],
                     simpleconstr[i][1], simpleconstr[i][2]),
                    file=f)

        if len(otherconstr) != 0:
            print('CONSTRAINTS', file=f)
            if self.constr_tol is None:
                print(len(otherconstr), file=f)
            else:
                print(len(otherconstr), utils.num2str(self.constr_tol), file=f)
            for x in otherconstr:
                print(x, file=f)

        if overridekpts is None:
            kp = self.kpts
        else:
            kp = overridekpts
        if kp == 'gamma':
            print('K_POINTS Gamma', file=f)
        else:
            x = np.shape(kp)
            if len(x) == 1:
                print('K_POINTS automatic', file=f)
                print(kp[0], kp[1], kp[2], end=' ', file=f)
                if overridekptshift is None:
                    print(
                        self.kptshift[0],
                        self.kptshift[1],
                        self.kptshift[2],
                        file=f)
                else:
                    print(
                        overridekptshift[0],
                        overridekptshift[1],
                        overridekptshift[2],
                        file=f)
            else:
                print('K_POINTS crystal', file=f)
                print(x[0], file=f)
                w = 1. / x[0]
                for k in kp:
                    if len(k) == 3:
                        print(
                            '%24.15e %24.15e %24.15e %24.15e' % (k[0], k[1],
                                                                 k[2], w),
                            file=f)
                    else:
                        print(
                            '%24.15e %24.15e %24.15e %24.15e' % (k[0], k[1],
                                                                 k[2], k[3]),
                            file=f)

        # closing PWscf input file ###
        f.close()
        if self.verbose == 'high':
            print('\nPWscf input file %s written\n' % fname)

    def set_atoms(self, atoms):
        if self.atoms is None or not self.started:
            self.atoms = atoms.copy()
        else:
            if len(atoms) != len(self.atoms):
                self.stop()
                self.nvalence = None
                self.nel = None
                self.recalculate = True

            x = atoms.cell - self.atoms.cell
            if np.max(x) > 1E-13 or np.min(x) < -1E-13:
                self.stop()
                self.recalculate = True
            if (atoms.get_atomic_numbers() !=
                    self.atoms.get_atomic_numbers()).any():
                self.stop()
                self.nvalence = None
                self.nel = None
                self.recalculate = True
            x = atoms.positions - self.atoms.positions
            if np.max(x) > 1E-13 or np.min(x) < - \
                    1E-13 or (not self.started and not self.got_energy):
                self.recalculate = True
        self.atoms = atoms.copy()

    def update(self, atoms):
        if self.atoms is None:
            self.set_atoms(atoms)
        x = atoms.cell - self.atoms.cell
        morethanposchange = np.max(x) > 1E-13 \
            or np.min(x) < -1E-13 \
            or len(atoms) != len(self.atoms) \
            or (atoms.get_atomic_numbers() 
                != self.atoms.get_atomic_numbers()
            ).any()
        x = atoms.positions - self.atoms.positions
        if np.max(x) > 1E-13 \
        or np.min(x) < -1E-13 \
        or morethanposchange \
        or (not self.started and not self.got_energy) \
        or self.recalculate:
            self.recalculate = True
            self.results = {}
            if self.ion_dynamics != 'ase3' \
            or self.calculation in ('scf','nscf') \
            or morethanposchange:
                self.stop()
            self.read(atoms)
        elif self.only_init:
            self.read(atoms)
        else:
            self.atoms = atoms.copy()

    def calculate(self, atoms=None, properties=['energy'],
                  system_changes=None):
        """
        ase 3.7+ compatibility method
        """
        if atoms is None:
            atoms = self.atoms
        self.update(atoms)

    def get_name(self):
        return 'QE-ASE3 interface'

    def get_version(self):
        return '0.1'

    def init_only(self, atoms):
        if self.atoms is None:
            self.set_atoms(atoms)
        x = atoms.positions - self.atoms.positions
        if (np.max(x) > 1E-13 or
            np.min(x) < - 1E-13 or
            (not self.started and not self.got_energy) or
            self.recalculate):
            self.recalculate = True
            if self.ion_dynamics != 'ase3':
                self.stop()

            if not self.started:
                self.initialize(atoms)
                self.only_init = True
            elif self.recalculate:
                self.only_init = True
                if self.ion_dynamics == 'ase3':
                    p = atoms.positions
                    self.atoms = atoms.copy()
                    self.cinp.write(b'G')
                    for x in p:
                        self.cinp.write(
                            ('%.15e %.15e %.15e' % (x[0], x[1], x[2])
                            ).replace('e', 'd').encode()
                        )
                self.cinp.flush()

    def read(self, atoms):
        if self.writeversion:
            self.writeversion = False
            with open(self.log, 'a') as s:
                s.write('  python dir          : ' + pkgpath + '\n')
                if len(self.exedir) == 0:
                    stdout = Popen(
                        'which pw.x', shell=True, stdout=PIPE).stdout
                    exedir = os.path.dirname(stdout.readline().decode('utf-8'))
                else:
                    exedir = self.exedir
                s.write('  espresso dir        : ' + str(exedir) + '\n')
                s.write('  pseudo dir          : ' + self.psppath + '\n')
                s.write('  ase-espresso py git : ' + gitver + '\n\n\n')

        if not self.started and not self.only_init:
            fresh = True
            self.initialize(atoms)
        else:
            fresh = False
        if self.recalculate:
            if not fresh and not self.only_init:
                if self.ion_dynamics == 'ase3':
                    p = atoms.positions
                    self.atoms = atoms.copy()
                    self.cinp.write(b'G')
                    for x in p:
                        self.cinp.write(
                            ('%.15e %.15e %.15e' % (x[0], x[1], x[2])
                            ).replace('e', 'd').encode()
                        )
                self.cinp.flush()
            self.only_init = False
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
            elif a == '' and self.calculation in ('ase3', 'relax', 'scf',
                                               'vc-relax', 'vc-md', 'md'):
                self.checkerror()
                # if checkerror shouldn't find an error here,
                # throw this generic error
                raise RuntimeError('SCF calculation failed')
            self.atom_occ = atom_occ
            self.results['magmoms'] = magmoms
            self.results['magmom'] = np.sum(magmoms)
            if self.calculation in ('ase3', 'relax', 'scf', 'vc-relax',
                                    'vc-md', 'md', 'hund'):
                self.energy_free = float(a.split()[-2]) * Rydberg
                # get S*T correction (there is none for Marzari-Vanderbilt=Cold
                # smearing)
                if (self.occupations == 'smearing' and
                    self.calculation != 'hund' and
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

            if self.calculation in ('ase3', 'relax', 'scf', 'vc-relax',
                                    'vc-md', 'md'):
                if self.ion_dynamics == 'ase3' and self.calculation != 'scf':
                    sys.stdout.flush()
                    while a[:5] != ' !ASE':
                        a = self.cout.readline().decode('utf-8')
                        s.write(a)
                        s.flush()
                    self.forces = np.empty((self.natoms, 3), np.float)
                    for i in range(self.natoms):
                        self.cout.readline().decode('utf-8')
                    for i in range(self.natoms):
                        self.forces[i][:] = [
                            float(x) for x in
                            self.cout.readline().decode('utf-8').split()]
                    self.forces *= (Rydberg / Bohr)
                else:
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

            # flush the rest of cout into the log file
            s.write(self.cout.read().decode('utf-8'))
            s.close()

            self.results['forces'] = self.forces
            if self.ion_dynamics != 'ase3':
                self.stop()

            # get final energy and forces for internal QE relaxation run
            if self.calculation in ('relax', 'vc-relax', 'vc-md', 'md'):
                if self.ion_dynamics == 'ase3':
                    self.stop()
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
                    self.calculation != 'hund' and
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
        """Create the pw.inp input file and start the calculation.
        If onlycreatepwinp is specified in calculator setup,
        only the input file will be written for manual submission.
        """
        if not self.started:
            self.atoms = atoms.copy()

            self.atoms2species()
            self.natoms = len(self.atoms)
            self.check_spinpol()
            if self.use_environ:
                self.writeenvinputfile()
            self.writeinputfile()
        if self.cancalc:
            self.start()

    def check_spinpol(self):
        mm = self.atoms.get_initial_magnetic_moments()
        sp = mm.any()
        self.summed_magmoms = np.sum(mm)
        if sp:
            if not self.spinpol and not self.noncollinear:
                raise KeyError(
                    'Explicitly specify spinpol=True or noncollinear=True '
                    'for spin-polarized systems'
                )
            elif abs(self.sigma) <= self.sigma_small and not self.fix_magmom:
                raise KeyError(
                    'Please use fix_magmom=True for sigma=0.0 eV and '
                    'spinpol=True. Hopefully this is not an extended '
                    'system...?'
                )
        else:
            if self.spinpol and abs(self.sigma) <= self.sigma_small:
                self.fix_magmom = True
        if abs(self.sigma) <= self.sigma_small:
            self.occupations = 'fixed'

    def start(self):
        if not self.started:
            if self.single_calculator:
                while len(espresso_calculators) > 0:
                    espresso_calculators.pop().stop()
                espresso_calculators.append(self)
            if self.site.batch:
                cdir = os.getcwd()
                os.chdir(self.localtmp)
                call(self.site.perHostMpiExec + ' cp ' + self.localtmp +
                     '/pw.inp ' + self.scratch, shell=True)
                if self.use_environ:
                    call(self.site.perHostMpiExec + ' cp ' + self.localtmp +
                         '/environ.in ' + self.scratch, shell=True)

                if self.calculation != 'hund':
                    #if not self.proclist:
                    self.cinp, self.cout = self.site.do_perProcMpiExec(
                        self.scratch, self.exedir + 'pw.x ' +
                        self.parflags + ' -in pw.inp')
                    #else:
                    #    (self.cinp,
                    #     self.cout,
                    #     self.cerr) = self.site.do_perSpecProcMpiExec(
                    #        self.mycpus, self.myncpus, self.scratch,
                    #        self.exedir + 'pw.x ' + self.parflags +
                    #        ' -in pw.inp|' + self.mypath + '/espfilter ' + str(
                    #            self.natoms) + ' ' + self.log + '0')
                else:
                    self.site.runonly_perProcMpiExec(
                        self.scratch, self.exedir + 'pw.x ' + self.serflags +
                        ' -in pw.inp >>' + self.log)
                    call(
                        "sed s/occupations.*/occupations=\\'fixed\\',/ <" +
                        self.localtmp +
                        "/pw.inp | sed s/ELECTRONS/ELECTRONS\\\\n\ \ startingwfc=\\'file\\',\\\\n\ \ startingpot=\\'file\\',/ | sed s/conv_thr.*/conv_thr="
                        + utils.num2str(self.conv_thr) +
                        ",/ | sed s/tot_magnetization.*/tot_magnetization=" +
                        utils.num2str(
                            self.totmag) + ",/ >" + self.localtmp + "/pw2.inp",
                        shell=True)
                    call(self.site.perHostMpiExec + ' cp ' + self.localtmp +
                         '/pw2.inp ' + self.scratch, shell=True)
                    if self.use_environ:
                        call(self.site.perHostMpiExec + ' cp ' + self.localtmp +
                             '/environ.in ' + self.scratch, shell=True)
                    self.cinp, self.cout = self.site.do_perProcMpiExec(
                        self.scratch,
                        self.exedir + 'pw.x ' + self.parflags + ' -in pw2.inp')
                os.chdir(cdir)
            else:
                call('cp ' + self.localtmp + '/pw.inp ' + self.scratch,
                     shell=True)
                if self.use_environ:
                    call('cp ' + self.localtmp + '/environ.in ' + self.scratch,
                         shell=True)
                if self.calculation != 'hund':
                    cmd = 'cd ' + self.scratch + ' ; ' + self.exedir + 'pw.x ' + self.serflags + ' -in pw.inp'
                    p = Popen(cmd, shell=True, stdin=PIPE,
                              stdout=PIPE, close_fds=True)
                    self.cinp, self.cout = (p.stdin, p.stdout)
                else:
                    call(
                        'cd ' + self.scratch + ' ; ' + self.exedir + 'pw.x ' +
                        self.serflags + ' -in pw.inp >>' + self.log, shell=True)
                    call(
                        "sed s/occupations.*/occupations=\\'fixed\\',/ <" +
                        self.localtmp +
                        "/pw.inp | sed s/ELECTRONS/ELECTRONS\\\\n\ \ startingwfc=\\'file\\',\\\\n\ \ startingpot=\\'file\\',/ | sed s/conv_thr.*/conv_thr="
                        + utils.num2str(self.conv_thr) +
                        ",/ | sed s/tot_magnetization.*/tot_magnetization=" +
                        utils.num2str(self.totmag) + ",/ >" + self.localtmp +
                        "/pw2.inp", shell=True
                    )
                    call('cp ' + self.localtmp + '/pw2.inp ' + self.scratch,
                         shell=True)
                    if self.use_environ:
                        call('cp ' + self.localtmp + '/environ.in ' +
                             self.scratch)

                    cmd = 'cd ' + self.scratch + ' ; ' + self.exedir + 'pw.x ' + self.serflags + ' -in pw2.inp'
                    p = Popen(cmd, shell=True, stdin=PIPE,
                              stdout=PIPE, close_fds=True)
                    self.cinp, self.cout = (p.stdin, p.stdout)

            self.started = True

    def stop(self):
        if self.started:
            if self.ion_dynamics == 'ase3':
                # sending 'Q' to espresso tells it to quit cleanly
                self.cinp.write(b'Q')
                try:
                    self.cinp.flush()
                except IOError:
                    # espresso may have already shut down, so flush may fail
                    pass
            else:
                self.cinp.flush()
            s = open(self.log, 'a')
            a = self.cout.readline().decode('utf-8')
            s.write(a)
            while a != '':
                a = self.cout.readline().decode('utf-8')
                s.write(a)
                s.flush()
            s.close()
            try:
                self.cinp.close()
            except:
                pass
            try:
                self.cout.close()
            except:
                pass
            self.started = False

    def topath(self, filename):
        if os.path.isabs(filename):
            return filename
        else:
            return os.path.join(self.sdir, filename)

    def save_output(self, filename='calc.tgz'):
        """
        Save the contents of calc.save directory.
        """
        self.topath(filename)
        self.update(self.atoms)
        self.stop()

        call('tar czf ' + filename + ' --directory=' + self.scratch +
             ' calc.save', shell=True)

    def load_output(self, filename='calc.tgz'):
        """
        Restore the contents of previously saved calc.save directory.
        """
        self.stop()
        self.topath(filename)

        call('tar xzf ' + filename + ' --directory=' + self.scratch,
             shell=True)

    def save_flev_output(self, filename='calc.tgz'):
        """
        Save the contents of calc.save directory + Fermi level
        & on-site density matrices (if present).
        """
        self.topath(filename)
        self.update(self.atoms)

        ef = self.get_fermi_level()
        f = open(self.scratch + '/calc.save/fermilevel.txt', 'w')
        print('%.15e\n#Fermi level in eV' % ef, file=f)
        f.close()

        call(
            'tar czf ' + filename + ' --directory=' + self.scratch +
            ' calc.save `find . -name "calc.occup*";find . -name "calc.paw"`',
            shell=True)

    def load_flev_output(self, filename='calc.tgz'):
        """
        Restore the contents of previously saved calc.save directory
        + Fermi level & on-site density matrices (if present).
        """
        self.stop()
        self.topath(filename)

        call('tar xzf ' + filename + ' --directory=' + self.scratch,
             shell=True)

        self.fermi_input = True
        with open(self.scratch + '/calc.save/fermilevel.txt', 'r') as f:
            self.inputfermilevel = float(f.readline())

    def save_chg(self, filename='chg.tgz'):
        """
        Save charge density.
        """
        self.topath(filename)
        self.update(self.atoms)
        self.stop()

        call(
            'tar czf ' + filename + ' --directory=' + self.scratch +
            ' calc.save/charge-density.dat calc.save/data-file.xml `cd ' +
            self.scratch +
            ';find calc.save -name "spin-polarization.*";find calc.save -name "magnetization.*";find . -name "calc.occup*";find . -name "calc.paw"`',
            shell=True
        )

    def load_chg(self, filename='chg.tgz'):
        """
        Load charge density.
        """
        self.stop()
        self.topath(filename)

        call('tar xzf ' + filename + ' --directory=' + self.scratch,
             shell=True)

    def save_wf(self, filename='wf.tgz'):
        """Save wave functions."""
        self.topath(filename)
        self.update(self.atoms)
        self.stop()

        call('tar czf ' + filename + ' --directory=' + self.scratch +
             ' --exclude=calc.save .', shell=True)

    def load_wf(self, filename='wf.tgz'):
        """Load wave functions."""
        self.stop()
        self.topath(filename)

        call('tar xzf ' + filename + ' --directory=' + self.scratch,
             shell=True)

    def save_flev_chg(self, filename='chg.tgz'):
        """
        Save charge density and Fermi level.
        Useful for subsequent bandstructure or density of states
        calculations.
        """
        self.topath(filename)
        self.update(self.atoms)

        ef = self.get_fermi_level()
        f = open(self.scratch + '/calc.save/fermilevel.txt', 'w')
        print('%.15e\n#Fermi level in eV' % ef, file=f)
        f.close()
        call(
            'tar czf ' + filename + ' --directory=' + self.scratch +
            ' calc.save/charge-density.dat calc.save/data-file.xml `cd ' +
            self.scratch +
            ';find calc.save -name "spin-polarization.*";find calc.save -name "magnetization.*";find . -name "calc.occup*";find . -name "calc.paw"` calc.save/fermilevel.txt',
            shell=True
        )

    def load_flev_chg(self, filename='efchg.tgz'):
        """
        Load charge density and Fermi level.
        Useful for subsequent bandstructure or density of states
        calculations.
        """
        self.stop()
        self.topath(filename)

        call('tar xzf ' + filename + ' --directory=' + self.scratch,
             shell=True)
        self.fermi_input = True
        f = open(self.scratch + '/calc.save/fermilevel.txt', 'r')
        self.inputfermilevel = float(f.readline().decode('utf-8'))
        f.close()

    def get_final_structure(self):
        """
        returns Atoms object according to a structure
        optimized internally by quantum espresso
        """
        self.stop()

        cmd = 'grep -a -n Giannozzi ' + self.log + '| tail -1'
        p = Popen(cmd, shell=True, stdout=PIPE).stdout
        n = int(p.readline().decode('utf-8').split()[0].strip(':'))

        s = open(self.log, 'r')
        # skip over previous runs in log in case the current log has been
        # appended to old ones
        for i in range(n):
            s.readline()

        a = s.readline()
        while a[:11] != '     celldm':
            a = s.readline()
        alat = float(a.split()[1]) / 1.889726
        a = s.readline()
        while a[:12] != '     crystal':
            a = s.readline()
        cell = []
        for i in range(3):
            cell.append([float(x) for x in
                         s.readline().split()[3:6]])
        cell = np.array(cell)
        a = s.readline()
        while a[:12] != '     site n.':
            a = s.readline()
        pos = []
        syms = ''
        y = s.readline().split()
        while len(y) > 0:
            nf = len(y)
            pos.append([float(x) for x in y[nf - 4:nf - 1]])
            syms += y[1].strip('0123456789')
            y = s.readline().split()
        pos = np.array(pos) * alat
        natoms = len(pos)

        # create atoms object with coordinates and unit cell
        # as specified in the initial ionic step in log
        atoms = Atoms(syms, pos, cell=cell * alat, pbc=(1, 1, 1))

        coord = 'angstrom)'
        a = s.readline()
        while a != '':
            while a[:7] != 'CELL_PA' and a[:7] != 'ATOMIC_' and a != '':
                a = s.readline()
            if a == '':
                break
            if a[0] == 'A':
                coord = a.split('(')[-1]
                for i in range(natoms):
                    pos[i][:] = s.readline().split()[1:4]
            else:
                for i in range(3):
                    cell[i][:] = s.readline().split()
            a = s.readline()

        atoms.set_cell(cell * alat, scale_atoms=False)

        if coord == 'alat)':
            atoms.set_positions(pos * alat)
        elif coord == 'bohr)':
            atoms.set_positions(pos * Bohr)
        elif coord == 'angstrom)':
            atoms.set_positions(pos)
        else:
            atoms.set_scaled_positions(pos)

        return atoms

    def get_potential_energy(self, atoms=None, force_consistent=False):
        self.update(atoms)
        if force_consistent:
            return self.energy_free
        else:
            return self.energy_zero

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

    def get_xc_functional(self):
        return self.xc

    def get_final_stress(self):
        """Returns 3x3 stress tensor after an internal
        unit cell relaxation in quantum espresso
        (also works for calcstress=True)
        """
        self.stop()

        cmd = 'grep -a -3 "total   stress" ' + self.log + ' | tail -3'
        p = Popen(cmd, shell=True, stdout=PIPE).stdout
        s = p.readlines()

        if len(s) != 3:
            raise RuntimeError(
                'Stress was not calculated\nconsider specifying '
                'calcstress or running a unit cell relaxation.'
            )

        stress = np.empty((3, 3), np.float)
        for i in range(3):
            stress[i][:] = [float(x) for x in s[i].split()[:3]]

        return stress * Rydberg / Bohr**3

    def get_stress(self, dummyself=None):
        """Returns stress tensor in Voigt notation """
        if self.calcstress:
            # ASE convention for the stress tensor appears to differ
            # from the PWscf one by a factor of -1
            stress = -1.0 * self.get_final_stress()
            # converting to Voigt notation as expected by ASE
            stress = np.array([
                stress[0, 0], stress[1, 1], stress[2, 2], stress[1, 2],
                stress[0, 2], stress[0, 1]
            ])
            self.results['stress'] = stress
            return stress
        else:
            raise NotImplementedError

    def get_magnetization(self):
        """Returns total and absolute magnetization after SCF run.
        Units are Bohr magnetons per unit cell, directly read PWscf log.
        Returns (0,0) if no magnetization is found in log.
        """
        cmd = 'grep -a "total magnetization" ' + self.log + ' | tail -1'
        p1 = Popen(cmd, shell=True, stdout=PIPE).stdout
        s1 = p1.readlines()

        cmd = 'grep -a "absolute magnetization" ' + self.log + ' | tail -1'
        p2 = Popen(cmd, shell=True, stdout=PIPE).stdout
        s2 = p2.readlines()

        if len(s1) == 0:
            assert len(s2) == 0
            return (0, 0)
        else:
            assert len(s1) == 1
            assert len(s2) == 1
            s1_ = s1[0].split("=")[-1]
            totmag = float(s1_.split("Bohr")[0])
            s2_ = s2[0].split("=")[-1]
            absmag = float(s2_.split("Bohr")[0])
            return (totmag, absmag)

    def get_smearing_contribution(self):
        return self.ST

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

    def relax_cell_and_atoms(
            self,
            cell_dynamics='bfgs',
            # {'none', 'sd', 'damp-pr', 'damp-w', 'bfgs'}
            ion_dynamics='bfgs',  # {'bfgs', 'damp'}
            cell_factor=1.2,
            cell_dofree=None,
            fmax=None,
            press=None,
            dpress=None):
        """Simultaneously relax unit cell and atoms using Espresso's internal
        relaxation routines.
        fmax,press are convergence limits and dpress is the convergence
        criterion wrt. reaching the target pressure press
        atoms.get_potential_energy() will yield the final energy,
        but to obtain the structure use
        relaxed_atoms = calc.get_final_structure()
        If you want to continue calculations in relax_atoms, use
        relaxed_atoms.set_calculator(some_espresso_calculator)
        """
        self.stop()
        oldmode = self.calculation
        oldalgo = self.ion_dynamics
        oldcell = self.cell_dynamics
        oldfactor = self.cell_factor
        oldfree = self.cell_dofree
        self.cell_dynamics = cell_dynamics
        self.ion_dynamics = ion_dynamics
        self.cell_factor = cell_factor
        self.cell_dofree = cell_dofree
        oldfmax = self.fmax
        oldpress = self.press
        olddpress = self.dpress

        if fmax is not None:
            self.fmax = fmax
        if press is not None:
            self.press = press
        if dpress is not None:
            self.dpress = dpress
        self.calculation = 'vc-relax'
        self.recalculate = True
        self.read(self.atoms)
        self.calculation = oldmode
        self.ion_dynamics = oldalgo
        self.cell_dynamics = oldcell
        self.cell_factor = oldfactor
        self.cell_dofree = oldfree
        self.fmax = oldfmax
        self.press = oldpress
        self.dpress = olddpress

    def relax_atoms(
            self,
            ion_dynamics='bfgs',  # {'bfgs', 'damp'}
            fmax=None):
        """Relax atoms using Espresso's internal relaxation routines.
        fmax is the force convergence limit
        atoms.get_potential_energy() will yield the final energy,
        but to obtain the structure use
        relaxed_atoms = calc.get_final_structure()
        If you want to continue calculations in relax_atoms, use
        relaxed_atoms.set_calculator(some_espresso_calculator)
        """
        self.stop()
        oldmode = self.calculation
        oldalgo = self.ion_dynamics
        self.ion_dynamics = ion_dynamics
        oldfmax = self.fmax

        self.calculation = 'relax'
        if fmax is not None:
            self.fmax = fmax
        self.recalculate = True
        self.read(self.atoms)
        self.calculation = oldmode
        self.ion_dynamics = oldalgo
        self.fmax = oldfmax

    def run_espressox(self,
                      binary,
                      inp,
                      log=None,
                      piperead=False,
                      parallel=True):
        """Runs one of the .x binaries of the espresso suite inp is expected to
        be in self.localtmp log will be created in self.localtmp
        """
        ll = ''
        if log is not None:
            ll += ' >> {}/{}'.format(self.localtmp, log)

        if self.site.batch and parallel:
            cdir = os.getcwd()
            os.chdir(self.localtmp)
            call(self.site.perHostMpiExec + ' cp ' + self.localtmp + '/' +
                 inp + ' ' + self.scratch, shell=True)
            if self.use_environ:
                call(self.site.perHostMpiExec + ' cp ' + self.localtmp +
                     '/environ.in' + ' ' + self.scratch, shell=True)

            if piperead:
                p = self.site.do_perProcMpiExec_outputonly(
                    self.scratch,
                    binary + ' ' + self.parflags + ' -in ' + inp + ll)
            else:
                self.site.runonly_perProcMpiExec(
                    self.scratch,
                    binary + ' ' + self.parflags + ' -in ' + inp + ll)
            os.chdir(cdir)
        else:
            if self.use_environ:
                call('cp {}/environ.in {}'.format(self.localtmp, self.scratch),
                     shell=True)

            call('cp {}/{} {}'.format(self.localtmp, inp, self.scratch),
                 shell=True)
            cmd = 'cd {} ; {} {} -in {}'.format(
                self.scratch, binary, self.serflags, inp + ll)
            if piperead:
                p = Popen(cmd, shell=True, stdout=PIPE).stdout
            else:
                call(cmd, shell=True)

        if piperead:
            return p

    def run_ppx(self,
                inp,
                log=None,
                inputpp=[],
                plot=[],
                output_format=5,
                iflag=3,
                piperead=False,
                parallel=True):
        if 'disk_io' in self.output:
            if self.output['disk_io'] == 'none':
                print(
                    "run_ppx requires output['disk_io'] to "
                    "be at least 'low' and avoidio=False"
                )
        self.stop()

        with open('{}/{}'.format(self.localtmp, inp), 'w') as f:
            f.write('&INPUTPP\n  prefix=\'calc\',\n  outdir=\'.\',\n')
            for a, b in inputpp:
                if isinstance(b, float):
                    c = utils.num2str(b)
                elif isinstance(b, str):
                    c = "'{}'".format(b)
                else:
                    c = str(b)
                f.write('  {}={},\n'.format(a, c))
            f.write('/\n')
            f.write('&PLOT\n  iflag={},\n  output_format={},\n'.format(
                iflag, output_format))
            for a, b in plot:
                if isinstance(b, float):
                    c = utils.num2str(b)
                elif isinstance(b, str):
                    c = "'{}'".format(b)
                else:
                    c = str(b)
                f.write('  {}={},\n'.format(a, c))
            f.write('/\n')

        if piperead:
            return self.run_espressox(
                self.exedir + 'pp.x',
                inp,
                log=log,
                piperead=piperead,
                parallel=parallel)
        else:
            self.run_espressox(
                self.exedir + 'pp.x', inp, log=log, parallel=parallel)

    def get_fermi_level(self):
        if self.fermi_input:
            return self.inputfermilevel
        self.stop()
        try:
            cmd = 'grep -a Fermi ' + self.log + '|tail -1'
            p = Popen(cmd, shell=True, stdout=PIPE).stdout
            efermi = float(p.readline().decode('utf-8').split()[-2])
        except BaseException:
            raise RuntimeError(
                'get_fermi_level called before DFT calculation was run')
        return efermi

    def calc_pdos(self,
                  Emin=None,
                  Emax=None,
                  DeltaE=None,
                  nscf=False,
                  tetrahedra=False,
                  slab=False,
                  kpts=None,
                  kptshift=None,
                  nbands=None,
                  ngauss=None,
                  sigma=None,
                  nscf_fermilevel=False,
                  add_higher_channels=True,
                  get_overlap_integrals=False):
        """Calculate (projected) density of states.
        - Emin,Emax,DeltaE define the energy window.
        - nscf=True will cause a non-selfconsistent calculation to be performed
          on top of a previous converged scf calculation, with the advantage
          that more kpts and more nbands can be defined improving the quality/
          increasing the energy range of the DOS.
        - tetrahedra=True (in addition to nscf=True) means use tetrahedron
          (i.e. smearing-free) method for DOS
        - slab=True: use triangle method insead of tetrahedron method
          (for 2D system perp. to z-direction)
        - sigma != None sets/overrides the smearing to calculate the DOS
          (also overrides tetrahedron/triangle settings)
        - get_overlap_integrals=True: also return k-point- and band-resolved
          projections (which are summed up and smeared to obtain the PDOS)

        Returns an array containing the energy window,
        the DOS over the same range,
        and the PDOS as an array (index: atom number 0..n-1) of dictionaries.
        The dictionary keys are the angular momentum channels 's','p','d'...
        (or e.g. 'p,j=0.5', 'p,j=1.5' in the case of LS-coupling).
        Each dictionary contains an array of arrays of the total and
        m-resolved PDOS over the energy window.
        In case of spin-polarization, total up is followed by total down, by
        first m with spin up, etc...

        Quantum Espresso with the tetrahedron method for PDOS can be
        obtained here:
        svn co --username anonymous http://qeforge.qe-forge.org/svn/q-e/branches/espresso-dynpy-beef
        """
        efermi = self.get_fermi_level()

        # run a nscf calculation with e.g. tetrahedra or more k-points etc.
        if nscf:
            if not hasattr(self, 'natoms'):
                self.atoms2species()
                self.natoms = len(self.atoms)
            if self.use_environ:
                self.writeenvinputfile()
            self.writeinputfile(
                filename='pwnscf.inp',
                mode='nscf',
                usetetrahedra=tetrahedra,
                overridekpts=kpts,
                overridekptshift=kptshift,
                overridenbands=nbands,
                suppressforcecalc=True)
            self.run_espressox(self.exedir + 'pw.x', 'pwnscf.inp',
                               'pwnscf.log')
            if nscf_fermilevel:
                cmd = 'grep -a Fermi ' + self.localtmp + '/pwnscf.log|tail -1'
                p = Popen(cmd, shell=True, stdout=PIPE).stdout
                efermi = float(p.readline().decode('utf-8').split()[-2])

        # remove old wave function projections
        call('rm -f ' + self.scratch + '/*_wfc*', shell=True)
        # create input for projwfc.x
        f = open(self.localtmp + '/pdos.inp', 'w')
        print('&PROJWFC\n  prefix=\'calc\',\n  outdir=\'.\',', file=f)
        if Emin is not None:
            print('  Emin = ' + utils.num2str(Emin + efermi) + ',', file=f)
        if Emax is not None:
            print('  Emax = ' + utils.num2str(Emax + efermi) + ',', file=f)
        if DeltaE is not None:
            print('  DeltaE = ' + utils.num2str(DeltaE) + ',', file=f)
        if slab:
            print('  lslab = .true.,', file=f)
        if ngauss is not None:
            print('  ngauss = ' + str(ngauss) + ',', file=f)
        if sigma is not None:
            print(
                '  degauss = ' + utils.num2str(sigma / Rydberg) + ',', file=f)
        print('/', file=f)
        f.close()
        # run projwfc.x
        self.run_espressox('projwfc.x', 'pdos.inp', 'pdos.log')

        # read in total density of states
        dos = np.loadtxt(self.scratch + '/calc.pdos_tot')
        if len(dos[0]) > 3:
            nspin = 2
            self.dos_total = [dos[:, 1], dos[:, 2]]
        else:
            nspin = 1
            self.dos_total = dos[:, 1]
        self.dos_energies = dos[:, 0] - efermi
        npoints = len(self.dos_energies)

        channels = {'s': 0, 'p': 1, 'd': 2, 'f': 3}
        # read in projections onto atomic orbitals
        self.pdos = [{} for i in range(self.natoms)]
        cmd = 'ls ' + self.scratch + '/calc.pdos_atm*'
        p = Popen(cmd, shell=True, stdout=PIPE).stdout
        proj = p.readlines()

        proj.sort()
        for i, inp in enumerate(proj):
            inpfile = inp.strip().decode('utf-8')
            pdosinp = np.genfromtxt(inpfile)
            spl = inpfile.split('#')
            iatom = int(spl[1].split('(')[0]) - 1
            channel = spl[2].split('(')[1].rstrip(')').replace('_j', ',j=')
            jpos = channel.find('j=')
            if jpos < 0:
                # ncomponents = 2*l+1 +1  (latter for m summed up)
                ncomponents = (2 * channels[channel[0]] + 2) * nspin
            else:
                # ncomponents = 2*j+1 +1  (latter for m summed up)
                ncomponents = int(2. * float(channel[jpos + 2:])) + 2
            if channel not in self.pdos[iatom]:
                self.pdos[iatom][channel] = np.zeros((ncomponents, npoints),
                                                     np.float)
                first = True
            else:
                first = False
            if add_higher_channels or first:
                for j in range(ncomponents):
                    self.pdos[iatom][channel][j] += pdosinp[:, (j + 1)]

        if get_overlap_integrals:
            return (self.dos_energies,
                    self.dos_total,
                    self.pdos,
                    self.__get_atomic_projections__())
        else:
            return self.dos_energies, self.dos_total, self.pdos

    def calc_bandstructure(self,
                           kptpath,
                           nbands=None,
                           atomic_projections=False):
        """Calculate bandstructure along kptpath (= array of k-points).
        If nbands is not None, override number of bands set in calculator.
        If atomic_projections is True, calculate orbital character of
        each band at each k-point.

        Returns an array of energies.
        (if spin-polarized spin is first index;
        the next index enumerates the k-points)
        """
        efermi = self.get_fermi_level()

        # run a nscf calculation
        if not hasattr(self, 'natoms'):
            self.atoms2species()
            self.natoms = len(self.atoms)
        oldnoinv = self.noinv
        oldnosym = self.nosym
        self.noinv = True
        self.nosym = True
        if self.use_environ:
            self.writeenvinputfile()
        self.writeinputfile(
            filename='pwnscf.inp',
            mode='nscf',
            overridekpts=kptpath,
            overridenbands=nbands,
            suppressforcecalc=True)
        self.noinv = oldnoinv
        self.nosym = oldnosym
        self.run_espressox(self.exedir + 'pw.x', 'pwnscf.inp', 'pwnscf.log')

        energies = self.get_eigenvalues(efermi=efermi)

        if not atomic_projections:
            return energies
        else:
            # run pdos calculation with (tiny) E-range
            # to trigger calculation of atomic_proj.xml

            # create input for projwfc.x
            f = open(self.localtmp + '/pdos.inp', 'w')
            print('&PROJWFC\n  prefix=\'calc\',\n  outdir=\'.\',', file=f)
            print('  filpdos = \'projtmp\',', file=f)
            print('  Emin = ' + utils.num2str(-0.3 + efermi) + ',', file=f)
            print('  Emax = ' + utils.num2str(-0.2 + efermi) + ',', file=f)
            print('  DeltaE = 0.1d0,', file=f)
            print('/', file=f)
            f.close()
            # run projwfc.x
            self.run_espressox('projwfc.x', 'pdos.inp', 'pdos.log')
            # remove unneeded pdos files containing only a tiny E-range of two
            # points
            call('rm -f ' + self.scratch + '/projtmp*', shell=True)

            return energies, self.__get_atomic_projections__()

    def __get_atomic_projections__(self):
        f = open(self.scratch + '/calc.save/atomic_proj.xml', 'r')
        cmd = 'grep -a -n Giannozzi ' + self.localtmp + '/pdos.log|tail -1'
        p = Popen(cmd, shell=True, stdout=PIPE).stdout
        n = p.readline().decode('utf-8').split()[0].strip(':').strip()

        cmd = 'tail -n +' + n + ' ' + self.localtmp + '/pdos.log|grep "state #"'
        p = Popen(cmd, shell=True, stdout=PIPE).stdout
        # identify states from projwfc.x's stdout
        states = []
        for x in p.readlines():
            y = x.split('atom')[1]
            iatom = int(y.split()[0]) - 1
            z = y.replace(')\n', '').split('=')
            if y.find('m_j') < 0:
                l = int(z[1].replace('m', ''))
                m = int(z[2])
                states.append([iatom, l, m])
            else:
                j = float(z[1].replace('l', ''))
                l = int(z[2].replace('m_j', ''))
                mj = float(z[3])
                states.append([iatom, j, l, mj])

        # read in projections from atomic_proj.xml
        a = f.readline().decode('utf-8')
        while a.find('<NUMBER_OF_B') < 0:
            a = f.readline().decode('utf-8')
        nbnd = int(f.readline().decode('utf-8').strip())
        a = f.readline().decode('utf-8')
        while a.find('<NUMBER_OF_K') < 0:
            a = f.readline().decode('utf-8')
        nkp = int(f.readline().decode('utf-8').strip())
        a = f.readline().decode('utf-8')
        while a.find('<NUMBER_OF_S') < 0:
            a = f.readline().decode('utf-8')
        spinpol = int(f.readline().decode('utf-8').strip()) == 2

        if spinpol:
            proj1 = []
            proj2 = []
            proj = proj1
        else:
            proj = []

        while a.find('<ATM') < 0 and a != '':
            a = f.readline().decode('utf-8')
        if a == '':
            raise RuntimeError('no projections found')

        while True:
            while a.find('<ATM') < 0 and a != '':
                if spinpol and a.find('<SP') >= 0:
                    if a.find('N.1') > 0:
                        proj = proj1
                    else:
                        proj = proj2
                a = f.readline().decode('utf-8')
            if a == '':
                break
            pr = np.empty(nbnd, np.complex)
            for i in range(nbnd):
                b = f.readline().decode('utf-8').split(',')
                pr[i] = float(b[0]) + 1j * float(b[1])
            proj.append(pr)
            a = f.readline().decode('utf-8')

        f.close()

        if spinpol:
            projections = np.array([proj1, proj2])
            return states, np.reshape(projections,
                                      (2, nkp, len(proj1) / nkp, nbnd))
        else:
            projections = np.array(proj)
            return states, np.reshape(projections,
                                      (nkp, len(proj) / nkp, nbnd))

    def get_eigenvalues(self, kpt=None, spin=None, efermi=None):
        self.stop()

        if self.spinpol:
            cmd = "grep -a eigenval1.xml " + self.scratch + "/calc.save/data-file.xml|tr '\"' ' '|awk '{print $(NF-1)}'"
            p = Popen(cmd, shell=True, stdout=PIPE).stdout
            kptdirs1 = [x.strip() for x in p.readlines()]

            kptdirs1.sort()
            cmd = "grep -a eigenval2.xml " + self.scratch + "/calc.save/data-file.xml|tr '\"' ' '|awk '{print $(NF-1)}'"
            p = Popen(cmd, shell=True, stdout=PIPE).stdout
            kptdirs2 = [x.strip() for x in p.readlines()]

            kptdirs2.sort()
            kptdirs = kptdirs1 + kptdirs2
        else:
            cmd = "grep -a eigenval.xml " + self.scratch + "/calc.save/data-file.xml|tr '\"' ' '|awk '{print $(NF-1)}'"
            p = Popen(cmd, shell=True, stdout=PIPE).stdout
            kptdirs = [x.strip() for x in p.readlines()]
            kptdirs.sort()

        nkp2 = len(kptdirs) / 2
        if kpt is None:  # get eigenvalues at all k-points
            if self.spinpol:
                if spin == 'up' or spin == 0:
                    kp = kptdirs[:nkp2]
                if spin == 'down' or spin == 1:
                    kp = kptdirs[nkp2:]
                else:
                    kp = kptdirs
            else:
                kp = kptdirs
        else:  # get eigenvalues at specific k-point
            if self.spinpol:
                if spin == 'up' or spin == 0:
                    kp = [kptdirs[kpt]]
                if spin == 'down' or spin == 1:
                    kp = [kptdirs[kpt + nkp2]]
                else:
                    kp = [kptdirs[kpt], kptdirs[kpt + nkp2 * 2]]
            else:
                kp = [kptdirs[kpt]]

        if efermi is None:
            ef = 0.
        else:
            ef = efermi

        eig = []
        for k in kp:
            f = open(self.scratch + '/calc.save/' + k, 'r')
            a = f.readline().decode('utf-8')
            while a.upper().find('<EIG') < 0:
                a = f.readline().decode('utf-8')
            nbnd = int(a.split('"')[-2])
            eig.append(Hartree * np.fromfile(
                f, dtype=float, count=nbnd, sep=' ') - ef)
            f.close()

        spinall = spin not in ('up', 'down', 0, 1)
        if kpt is not None and spinall:
            return np.array(eig[0])
        elif kpt is None and spinall and self.spinpol:
            return np.reshape(np.array(eig), (2, nkp2, nbnd))
        else:
            return np.array(eig)

    def read_3d_grid(self, stream, log):
        with open(self.localtmp + '/' + log, 'a') as f:
            x = stream.readline().decode('utf-8')
            while x != '' and x[:11] != 'DATAGRID_3D':
                f.write(x)
                x = stream.readline().decode('utf-8')
            if x == '':
                raise RuntimeError('error reading 3D data grid')
            f.write(x)

            n = [int(_) for _ in stream.readline().split()]
            origin = np.array([float(_) for _ in stream.readline().split()])

            cell = np.empty((3, 3))
            for i in range(3):
                cell[i][:] = [float(_) for _ in stream.readline().split()]

            data = []
            ntotal = np.prod(n)
            while len(data) < ntotal:
                data += [float(_) for _ in stream.readline().split()]
            data = np.reshape(data, n, order='F')

            x = stream.readline().decode('utf-8')
            while x != '':
                f.write(x)
                x = stream.readline().decode('utf-8')

        return (origin, cell, data)

    def extract_charge_density(self, spin='both'):
        """
        Obtains the charge density as a numpy array after a DFT calculation.
        Returns (origin,cell,density).
        """
        if spin == 'both' or spin == 0:
            s = 0
        elif spin == 'up' or spin == 1:
            s = 1
        elif spin == 'down' or spin == 2:
            s = 2
        else:
            raise ValueError('unknown spin component')

        p = self.run_ppx(
            'charge.inp',
            inputpp=[['plot_num', 0], ['spin_component', s]],
            piperead=True,
            parallel=False)
        origin, cell, data = self.read_3d_grid(p, 'charge.log')
        p.close()
        return (origin, cell, data)

    def xsf_charge_density(self, xsf, spin='both'):
        """
        Writes the charge density from a DFT calculation
        to an input file for xcrysden.
        """
        if spin == 'both' or spin == 0:
            s = 0
        elif spin == 'up' or spin == 1:
            s = 1
        elif spin == 'down' or spin == 2:
            s = 2
        else:
            raise ValueError('unknown spin component')

        self.run_ppx(
            'charge.inp',
            inputpp=[['plot_num', 0], ['spin_component', s]],
            plot=[['fileout', self.topath(xsf)]],
            parallel=False,
            log='charge.log')

    def extract_total_potential(self, spin='both'):
        """
        Obtains the total potential as a numpy array after a DFT calculation.
        Returns (origin,cell,potential).
        """
        if spin == 'both' or spin == 0:
            s = 0
        elif spin == 'up' or spin == 1:
            s = 1
        elif spin == 'down' or spin == 2:
            s = 2
        else:
            raise ValueError('unknown spin component')

        p = self.run_ppx(
            'totalpot.inp',
            inputpp=[['plot_num', 1], ['spin_component', s]],
            piperead=True,
            parallel=False)
        origin, cell, data = self.read_3d_grid(p, 'totalpot.log')
        p.close()
        return (origin, cell, data * Rydberg)

    def xsf_total_potential(self, xsf, spin='both'):
        """
        Writes the total potential from a DFT calculation
        to an input file for xcrysden.
        """
        if spin == 'both' or spin == 0:
            s = 0
        elif spin == 'up' or spin == 1:
            s = 1
        elif spin == 'down' or spin == 2:
            s = 2
        else:
            raise ValueError('unknown spin component')

        self.run_ppx(
            'totalpot.inp',
            inputpp=[['plot_num', 1], ['spin_component', s]],
            plot=[['fileout', self.topath(xsf)]],
            parallel=False,
            log='totalpot.log')

    def extract_local_ionic_potential(self):
        """Obtains the local ionic potential as a numpy array after a DFT calculation.
        Returns (origin,cell,potential).
        """
        p = self.run_ppx(
            'vbare.inp',
            inputpp=[['plot_num', 2]],
            piperead=True,
            parallel=False)
        origin, cell, data = self.read_3d_grid(p, 'vbare.log')
        p.close()
        return (origin, cell, data * Rydberg)

    def xsf_local_ionic_potential(self, xsf):
        """
        Writes the local ionic potential from a DFT calculation
        to an input file for xcrysden.
        """
        self.run_ppx(
            'vbare.inp',
            inputpp=[['plot_num', 2]],
            plot=[['fileout', self.topath(xsf)]],
            parallel=False,
            log='vbare.log')

    def extract_local_dos_at_efermi(self):
        """Obtains the local DOS at the Fermi level as a numpy array
        after a DFT calculation. Returns (origin,cell,ldos).
        """
        p = self.run_ppx(
            'ldosef.inp',
            inputpp=[['plot_num', 3]],
            piperead=True,
            parallel=False)
        origin, cell, data = self.read_3d_grid(p, 'ldosef.log')
        p.close()
        return (origin, cell, data)

    def xsf_local_dos_at_efermi(self, xsf):
        """
        Writes the local DOS at the Fermi level from a DFT calculation
        to an input file for xcrysden.
        """
        self.run_ppx(
            'ldosef.inp',
            inputpp=[['plot_num', 3]],
            plot=[['fileout', self.topath(xsf)]],
            parallel=False,
            log='ldosef.log')

    def extract_local_entropy_density(self):
        """Obtains the local entropy density as a numpy array after a
        DFT calculation. Returns (origin,cell,density).
        """
        p = self.run_ppx(
            'lentr.inp',
            inputpp=[['plot_num', 4]],
            piperead=True,
            parallel=False)
        origin, cell, data = self.read_3d_grid(p, 'lentr.log')
        p.close()
        return (origin, cell, data)

    def xsf_local_entropy_density(self, xsf):
        """
        Writes the local entropy density from a DFT calculation
        to an input file for xcrysden.
        """
        self.run_ppx(
            'lentr.inp',
            inputpp=[['plot_num', 4]],
            plot=[['fileout', self.topath(xsf)]],
            parallel=False,
            log='lentr.log')

    def extract_stm_data(self, bias):
        """
        Obtains STM data as a numpy array after a DFT calculation.
        Returns (origin,cell,stmdata).
        """
        p = self.run_ppx(
            'stm.inp',
            inputpp=[['plot_num', 5], ['sample_bias', bias / Rydberg]],
            piperead=True,
            parallel=False)
        origin, cell, data = self.read_3d_grid(p, 'stm.log')
        p.close()
        return (origin, cell, data)

    def xsf_stm_data(self, xsf, bias):
        """
        Writes STM data from a DFT calculation
        to an input file for xcrysden.
        """
        self.run_ppx(
            'stm.inp',
            inputpp=[['plot_num', 5], ['sample_bias', bias / Rydberg]],
            plot=[['fileout', self.topath(xsf)]],
            parallel=False,
            log='stm.log')

    def extract_magnetization_density(self):
        """Obtains the magnetization density as a numpy array after a
        DFT calculation. Returns (origin,cell,density).
        """
        p = self.run_ppx(
            'magdens.inp',
            inputpp=[['plot_num', 6]],
            piperead=True,
            parallel=False)
        origin, cell, data = self.read_3d_grid(p, 'magdens.log')
        p.close()
        return (origin, cell, data)

    def xsf_magnetization_density(self, xsf):
        """
        Writes the magnetization density from a DFT calculation
        to an input file for xcrysden.
        """
        self.run_ppx(
            'magdens.inp',
            inputpp=[['plot_num', 6]],
            plot=[['fileout', self.topath(xsf)]],
            parallel=False,
            log='magdens.log')

    def extract_wavefunction_density(self,
                                     band,
                                     kpoint=0,
                                     spin='up',
                                     gamma_with_sign=False):
        """Obtains the amplitude of a given wave function as a numpy array after
        a DFT calculation. Returns (origin,cell,amplitude).
        """
        if spin == 'up' or spin == 1:
            s = 0
        elif spin == 'down' or spin == 2:
            s = 1
        elif spin == 'charge' or spin == 0:
            s = 0
        elif spin == 'x':
            s = 1
        elif spin == 'y':
            s = 2
        elif spin == 'z':
            s = 3
        else:
            raise ValueError('unknown spin component')
        if self.spinpol:
            cmd = 'grep -a "number of k points=" ' + self.log + '|tail -1|tr \'=\' \' \''
            p = Popen(cmd, shell=True, stdout=PIPE).stdout
            nkp = int(p.readline().decode('utf-8').split()[4])
            kp = kpoint + nkp / 2 * s
        else:
            kp = kpoint
        inputpp = [['plot_num', 7], ['kpoint', kp], ['kband', band]]
        if gamma_with_sign:
            inputpp.append(['lsign', '.true.'])
        if self.noncollinear:
            inputpp.append(['spin_component', s])
        p = self.run_ppx(
            'wfdens.inp', inputpp=inputpp, piperead=True, parallel=True)
        origin, cell, data = self.read_3d_grid(p, 'wfdens.log')
        p.close()
        return (origin, cell, data)

    def xsf_wavefunction_density(self,
                                 xsf,
                                 band,
                                 kpoint=0,
                                 spin='up',
                                 gamma_with_sign=False):
        """
        Writes the amplitude of a given wave function from a DFT calculation
        to an input file for xcrysden.
        """
        if spin == 'up' or spin == 1:
            s = 0
        elif spin == 'down' or spin == 2:
            s = 1
        elif spin == 'charge' or spin == 0:
            s = 0
        elif spin == 'x':
            s = 1
        elif spin == 'y':
            s = 2
        elif spin == 'z':
            s = 3
        else:
            raise ValueError('unknown spin component')
        if self.spinpol:
            cmd = 'grep -a "number of k points=" ' + self.log + '|tail -1|tr \'=\' \' \''
            p = Popen(cmd, shell=True, stdout=PIPE).stdout
            nkp = int(p.readline().decode('utf-8').split()[4])
            kp = kpoint + nkp / 2 * s
        else:
            kp = kpoint
        inputpp = [['plot_num', 7], ['kpoint', kp], ['kband', band]]
        if gamma_with_sign:
            inputpp.append(['lsign', '.true.'])
        if self.noncollinear:
            inputpp.append(['spin_component', s])
        self.run_ppx(
            'wfdens.inp',
            inputpp=inputpp,
            plot=[['fileout', self.topath(xsf)]],
            parallel=True,
            log='wfdens.log')

    def extract_electron_localization_function(self):
        """
        Obtains the ELF as a numpy array after a DFT calculation.
        Returns (origin,cell,elf).
        """
        p = self.run_ppx(
            'elf.inp',
            inputpp=[['plot_num', 8]],
            piperead=True,
            parallel=False)
        origin, cell, data = self.read_3d_grid(p, 'elf.log')
        p.close()
        return (origin, cell, data)

    def xsf_electron_localization_function(self, xsf):
        """
        Writes the ELF from a DFT calculation
        to an input file for xcrysden.
        """
        self.run_ppx(
            'elf.inp',
            inputpp=[['plot_num', 8]],
            plot=[['fileout', self.topath(xsf)]],
            parallel=False,
            log='elf.log')

    def extract_density_minus_atomic(self):
        """Obtains the charge density minus atomic charges as a numpy array
        after a DFT calculation. Returns (origin,cell,density).
        """
        p = self.run_ppx(
            'dens_wo_atm.inp',
            inputpp=[['plot_num', 9]],
            piperead=True,
            parallel=False)
        origin, cell, data = self.read_3d_grid(p, 'dens_wo_atm.log')
        p.close()
        return (origin, cell, data)

    def xsf_density_minus_atomic(self, xsf):
        """
        Writes the charge density minus atomic charges from a DFT calculation
        to an input file for xcrysden.
        """
        self.run_ppx(
            'dens_wo_atm.inp',
            inputpp=[['plot_num', 9]],
            plot=[['fileout', self.topath(xsf)]],
            parallel=False,
            log='dens_wo_atm.log')

    def extract_int_local_dos(self, spin='both', emin=None, emax=None):
        """
        Obtains the integrated ldos as a numpy array after a DFT calculation.
        Returns (origin,cell,ldos).
        """
        if spin == 'both' or spin == 0:
            s = 0
        elif spin == 'up' or spin == 1:
            s = 1
        elif spin == 'down' or spin == 2:
            s = 2
        else:
            raise ValueError('unknown spin component')

        inputpp = [['plot_num', 10], ['spin_component', s]]
        efermi = self.get_fermi_level()
        if emin is not None:
            inputpp.append(['emin', emin - efermi])
        if emax is not None:
            inputpp.append(['emax', emax - efermi])

        p = self.run_ppx(
            'ildos.inp', inputpp=inputpp, piperead=True, parallel=False)
        origin, cell, data = self.read_3d_grid(p, 'ildos.log')
        p.close()
        return (origin, cell, data)

    def xsf_int_local_dos(self, xsf, spin='both', emin=None, emax=None):
        """
        Writes the integrated ldos from a DFT calculation
        to an input file for xcrysden.
        """
        if spin == 'both' or spin == 0:
            s = 0
        elif spin == 'up' or spin == 1:
            s = 1
        elif spin == 'down' or spin == 2:
            s = 2
        else:
            raise ValueError('unknown spin component')

        inputpp = [['plot_num', 10], ['spin_component', s]]
        efermi = self.get_fermi_level()
        if emin is not None:
            inputpp.append(['emin', emin - efermi])
        if emax is not None:
            inputpp.append(['emax', emax - efermi])

        self.run_ppx(
            'ildos.inp',
            inputpp=inputpp,
            plot=[['fileout', self.topath(xsf)]],
            parallel=False,
            log='ildos.log')

    def extract_ionic_and_hartree_potential(self):
        """Obtains the sum of ionic and Hartree potential as a numpy array
        after a DFT calculation. Returns (origin,cell,potential).
        """
        p = self.run_ppx(
            'potih.inp',
            inputpp=[['plot_num', 11]],
            piperead=True,
            parallel=False)
        origin, cell, data = self.read_3d_grid(p, 'potih.log')
        p.close()
        return (origin, cell, data * Rydberg)

    def xsf_ionic_and_hartree_potential(self, xsf):
        """
        Writes the sum of ionic and Hartree potential from a DFT calculation
        to an input file for xcrysden.
        """
        self.run_ppx(
            'potih.inp',
            inputpp=[['plot_num', 11]],
            plot=[['fileout', self.topath(xsf)]],
            parallel=False,
            log='potih.log')

    def extract_sawtooth_potential(self):
        """Obtains the saw tooth potential as a numpy array
        after a DFT calculation. Returns (origin,cell,potential).
        """
        p = self.run_ppx(
            'sawtooth.inp',
            inputpp=[['plot_num', 12]],
            piperead=True,
            parallel=False)
        origin, cell, data = self.read_3d_grid(p, 'sawtooth.log')
        p.close()
        return (origin, cell, data * Rydberg)

    def xsf_sawtooth_potential(self, xsf):
        """
        Writes the saw tooth potential from a DFT calculation
        to an input file for xcrysden.
        """
        self.run_ppx(
            'sawtooth.inp',
            inputpp=[['plot_num', 12]],
            plot=[['fileout', self.topath(xsf)]],
            parallel=False,
            log='sawtooth.log')

    def extract_noncollinear_magnetization(self, spin='all'):
        """Obtains the non-collinear magnetization as a numpy array
        after a DFT calculation. Returns (origin,cell,magnetization).
        """
        if spin == 'all' or spin == 'charge' or spin == 0:
            s = 0
        elif spin == 'x':
            s = 1
        elif spin == 'y':
            s = 2
        elif spin == 'z':
            s = 3
        else:
            raise ValueError('unknown spin component')
        p = self.run_ppx(
            'noncollmag.inp',
            inputpp=[['plot_num', 13], ['spin_component', s]],
            piperead=True,
            parallel=False)
        origin, cell, data = self.read_3d_grid(p, 'noncollmag.log')
        p.close()
        return (origin, cell, data)

    def xsf_noncollinear_magnetization(self, xsf, spin='all'):
        """
        Writes the non-collinear magnetization as from a DFT calculation
        to an input file for xcrysden.
        """
        if spin == 'all' or spin == 'charge' or spin == 0:
            s = 0
        elif spin == 'x':
            s = 1
        elif spin == 'y':
            s = 2
        elif spin == 'z':
            s = 3
        else:
            raise ValueError('unknown spin component')
        self.run_ppx(
            'noncollmag.inp',
            inputpp=[['plot_num', 13], ['spin_component', s]],
            plot=[['fileout', self.topath(xsf)]],
            parallel=False)

    def extract_ae_charge_density(self, spin='both'):
        """Obtains the all-electron (PAW) charge density as a numpy array
        after a DFT calculation. Returns (origin,cell,density)
        """
        if spin == 'both' or spin == 0:
            s = 0
        elif spin == 'up' or spin == 1:
            s = 1
        elif spin == 'down' or spin == 2:
            s = 2
        else:
            raise ValueError('unknown spin component')

        p = self.run_ppx(
            'aecharge.inp',
            inputpp=[['plot_num', 17], ['spin_component', s]],
            piperead=True,
            parallel=False)
        origin, cell, data = self.read_3d_grid(p, 'aecharge.log')
        p.close()
        return (origin, cell, data)

    def xsf_ae_charge_density(self, xsf, spin='both'):
        """
        Writes the all-electron (PAW) charge density from a DFT calculation
        to an input file for xcrysden.
        """
        if spin == 'both' or spin == 0:
            s = 0
        elif spin == 'up' or spin == 1:
            s = 1
        elif spin == 'down' or spin == 2:
            s = 2
        else:
            raise ValueError('unknown spin component')

        self.run_ppx(
            'aecharge.inp',
            inputpp=[['plot_num', 17], ['spin_component', s]],
            plot=[['fileout', self.topath(xsf)]],
            parallel=False,
            log='aecharge.log')

    def extract_noncollinear_xcmag(self):
        """Obtains the xc magnetic field for a non-collinear system as a numpy array
        after a DFT calculation. Returns (origin,cell,field).
        """
        p = self.run_ppx(
            'ncxcmag.inp',
            inputpp=[['plot_num', 18]],
            piperead=True,
            parallel=False)
        origin, cell, data = self.read_3d_grid(p, 'ncxcmag.log')
        p.close()
        return (origin, cell, data)

    def xsf_noncollinear_xcmag(self, xsf):
        """ Writes the xc magnetic field for a non-collinear system from
        a DFT calculation to an input file for xcrysden.
        """
        self.run_ppx(
            'ncxcmag.inp',
            inputpp=[['plot_num', 18]],
            plot=[['fileout', self.topath(xsf)]],
            parallel=False,
            log='ncxcmag.log')

    def extract_reduced_density_gradient(self):
        """Obtains the reduced density gradient as a numpy array after
        a DFT calculation. Returns (origin,cell,gradient).
        """
        p = self.run_ppx(
            'redgrad.inp',
            inputpp=[['plot_num', 19]],
            piperead=True,
            parallel=False)
        origin, cell, data = self.read_3d_grid(p, 'redgrad.log')
        p.close()
        return (origin, cell, data)

    def xsf_reduced_density_gradient(self, xsf):
        """
        Writes the reduced density gradient from a DFT calculation
        to an input file for xcrysden.
        """
        self.run_ppx(
            'redgrad.inp',
            inputpp=[['plot_num', 19]],
            plot=[['fileout', self.topath(xsf)]],
            parallel=False,
            log='redgrad.log')

    def extract_middle_density_hessian_eig(self):
        """Obtains the middle Hessian eigenvalue as a numpy array after
        a DFT calculation. Returns (origin,cell,density).
        """
        p = self.run_ppx(
            'mideig.inp',
            inputpp=[['plot_num', 20]],
            piperead=True,
            parallel=False)
        origin, cell, data = self.read_3d_grid(p, 'mideig.log')
        p.close()
        return (origin, cell, data)

    def xsf_middle_density_hessian_eig(self, xsf):
        """Writes the middle Hessian eigenvalue from a DFT calculation
        to an input file for xcrysden.
        """
        self.run_ppx(
            'mideig.inp',
            inputpp=[['plot_num', 20]],
            plot=[['fileout', self.topath(xsf)]],
            parallel=False,
            log='mideig.log')

    def find_max_empty_space(self, edir=3):
        """Assuming periodic boundary conditions, finds the largest
        continuous segment of free, unoccupied space and returns
        its midpoint in scaled coordinates (0 to 1) in the edir
        direction (default z).
        """
        position_array = self.atoms.get_scaled_positions()[
            ..., edir - 1]  # 0-indexed direction
        position_array.sort()
        differences = np.diff(position_array)
        differences = np.append(
            differences,
            position_array[0] + 1 - position_array[-1])  # through the PBC
        max_diff_index = np.argmax(differences)
        if max_diff_index == len(position_array) - 1:
            # should be < 1 in cell units
            return (position_array[0] + 1 + position_array[-1]) / 2. % 1
        else:
            return (position_array[max_diff_index] +
                    position_array[max_diff_index + 1]) / 2.

    def get_work_function(self, pot_filename="pot.xsf", edir=3):
        """Calculates the work function of a calculation by subtracting
        the electrostatic potential of the vacuum (from averaging the
        output of pp.x num_plot 11 in the z direction by default) from
        the Fermi energy. Values used for average.x come from the espresso
        example for work function for a surface.
        """
        self.update(self.atoms)
        self.stop()
        self.run_ppx(
            'wf_pp.in',
            log='wf_pp.log',
            inputpp=[('plot_num', 11), ('filplot', self.topath('pot.xsf'))],
            output_format=3,
            iflag=3,
            piperead=False,
            parallel=False)

        f = open(self.localtmp + '/avg.in', 'w')
        print('1', file=f)
        print(self.sdir + "/" + pot_filename, file=f)
        print('1.D0', file=f)
        print('1440', file=f)
        print(str(edir), file=f)
        print('3.835000000', file=f)
        print('', file=f)
        f.close()
        call('cp ' + self.localtmp + '/avg.in ' + self.scratch, shell=True)
        call('cd ' + self.scratch + ' ; ' + 'average.x < avg.in >>' +
             self.localtmp + '/avg.out', shell=True)
        call('cp ' + self.scratch + '/avg.dat ' + self.localtmp, shell=True)

        # Pick a good place to sample vacuum level
        cell_length = self.atoms.cell[edir - 1][edir - 1] / Bohr
        vacuum_pos = self.find_max_empty_space(edir) * cell_length
        avg_out = open(self.localtmp + '/avg.dat', 'r')
        record = False
        average_data = []
        lines = list(avg_out)
        for line in lines:
            if len(line.split()) == 3 and line.split()[0] == "0.000000000":
                record = True
            elif len(line.split()) == 0:
                record = False
            if record:
                average_data.append([float(i) for i in line.split()])
        # [1] is planar average [2] is macroscopic average
        vacuum_energy = average_data[np.abs(
            np.array(average_data)[..., 0] - vacuum_pos).argmin()][1]

        # Get the latest Fermi energy
        cmd = 'grep -a -n "Fermi" ' + self.log + ' | tail -1'
        fermi_data = Popen(cmd, shell=True, stdout=PIPE).stdout
        fermi_energy = float(fermi_data.readline().decode('utf-8').split()[-2])
        fermi_data.close()

        # if there's a dipole, we need to return 2 work functions - one for
        # either direction away from the slab
        if self.dipole['status']:
            eopreg = 0.025
            if 'eopreg' in self.dipole:
                eopreg = self.dipole['eopreg']
            # we use cell_length*eopreg*2.5 here since the work functions seem
            # to converge at that distance rather than *1 or *2
            vac_pos1 = (vacuum_pos - cell_length * eopreg * 2.5) % cell_length
            vac_pos2 = (vacuum_pos + cell_length * eopreg * 2.5) % cell_length
            vac_index1 = np.abs(
                np.array(average_data)[..., 0] - vac_pos1).argmin()
            vac_index2 = np.abs(
                np.array(average_data)[..., 0] - vac_pos2).argmin()
            vacuum_energy1 = average_data[vac_index1][1]
            vacuum_energy2 = average_data[vac_index2][1]
            wf = [
                vacuum_energy1 * Rydberg - fermi_energy,
                vacuum_energy2 * Rydberg - fermi_energy
            ]
        else:
            wf = vacuum_energy * Rydberg - fermi_energy

        return wf

    def generate_dummy_data(self):
        """Generate calc.save/data-file.xml, with non-sense electronic dispersion
        data (1-kpoint and 1 unconverged band), to be able to extract
        charge-density-only-dependent output data in case only the
        charge-density was stored.
        """
        convsave = self.convergence.copy()
        occupationssave = self.occupations
        self.occupations = 'fixed'
        # avoid espresso performing diagonalization
        self.convergence = {
            'maxsteps': -1,
            'diag': 'cg',
            'diago_cg_max_iter': -1,
            'energy': 1e80
        }
        if not hasattr(self, 'natoms'):
            self.atoms2species()
            self.natoms = len(self.atoms)
        if self.use_environ:
            self.writeenvinputfile()
        self.writeinputfile(
            filename='nonsense.inp',
            mode='nscf',
            overridekpts=(1, 1, 1),
            overridekptshift=(0, 0, 0),
            overridenbands=1,
            suppressforcecalc=True)
        self.run_espressox(
            self.exedir + 'pw.x',
            'nonsense.inp',
            'nonsense.log',
            parallel=False)
        self.occupations = occupationssave
        del self.convergence
        self.convergence = convsave

    def get_world(self):
        return world(self.site.nprocs)

    def get_number_of_scf_steps(self, all=False):
        """Get number of steps for convered scf. Returns an array.
        Option 'all' gives all numbers of steps in log,
        not only for the latest scf."""
        if all:
            tail = 'tail'
        else:
            tail = 'tail -1'
        cmd = 'grep -a "convergence has been achieved in" ' + self.log + ' | ' + tail
        p = Popen(cmd, shell=True, stdout=PIPE).stdout
        s = p.readlines()

        if not all:
            assert len(s) < 2
        if len(s) == 0:
            return None
        else:
            out = []
            for s_ in s:
                tmp = s_.split('in')
                out.append(int(tmp[-1].split('iterations')[0]))
            return out

    def get_number_of_bfgs_steps(self):
        """Get total number of internal BFGS steps."""
        cmd = 'grep -a "bfgs converged in" ' + self.log + ' | tail -1'
        p = Popen(cmd, shell=True, stdout=PIPE).stdout
        s = p.readlines()
        assert len(s) < 2
        if len(s) == 0:
            return None
        else:
            tmp = s[0].split('and')
            return int(tmp[-1].split('bfgs')[0])

    def get_forces(self, atoms):
        self.update(atoms)
        if self.newforcearray:
            return self.forces.copy()
        else:
            return self.forces
