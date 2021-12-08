# Table of contents
1. [Overview of the Cobaya-CosmoLike Joint Architecture (Cocoa)](#overview)
2. [Installation of Cocoa's required packages](#required_packages)
    1. [Via Conda (best for Linux)](#required_packages_conda)
    2. [Via Docker (best for MacOS/Windows)](#required_packages_docker)
    3. [(expert) Via Cocoa's internal cache](#required_packages_cache)
3. [Installation of Cobaya base code](#cobaya_base_code)
4. [Running Cobaya Examples](#cobaya_base_code_examples)
6. [Running Cosmolike projects](#running_cosmolike_projects)
7. [Creating Cosmolike projects](#creating_cosmolike_projects)
8. [Appendix](#appendix)
    1. [Proper Credits](#appendix_proper_credits)
    1. [Compiling Boltzmann, CosmoLike and Likelihood codes separatelly](#appendix_compile_separatelly)
    2. [Running Jupyter Notebooks inside the Whovian-Cosmo docker container](#appendix_jupyter_whovian)
    3. [Summary Information about Cocoa's configuration files](#appendix_config_files)

## Overview of the [Cobaya](https://github.com/CobayaSampler)-[CosmoLike](https://github.com/CosmoLike) Joint Architecture (Cocoa) <a name="overview"></a>

Cocoa allows users to run [CosmoLike](https://github.com/CosmoLike) routines inside the [Cobaya](https://github.com/CobayaSampler) framework. Cosmolike is capable of analyzing data primarily from the [Dark Energy Survey](https://www.darkenergysurvey.org) (a.k.a DES) and simulating future multi-probe analyses for Rubin Observatory's Legacy Survey of Space and Time or the Roman Space Telescope. This readme file presents basic and advanced instructions for installing all Cocoa components, including the [Planck likelihood](https://wiki.cosmos.esa.int/planck-legacy-archive/index.php/Main_Page). 

## Installation of Cocoa's required packages <a name="required_packages"></a>

Cosmolike, including the interface between Cosmolike and Cobaya, requires some C, C++ and Python packages to be installed. These packages include [GSL](https://www.gnu.org/software/gsl/), [FFTW](https://www.fftw.org), [Armadillo](http://arma.sourceforge.net), [Pybind11](https://pybind11.readthedocs.io/en/stable/) and [Boost](https://www.boost.org). In addition, Planck likelihood requires [CFITSIO](https://heasarc.gsfc.nasa.gov/fitsio/), and Cobaya requires many additional Python packages. The overabundance of compiler and package versions, each one with a different set of bugs and regressions, can make the installation of any big code (and the verification of numerical results) to be pure agony. In this section, we try to simplify installation, and more importantly, **standardize the package environment** adopted by Cocoa.

### Via Conda (best for Linux/HPC) <a name="required_packages_conda"></a>

The simpler way to install most prerequisites is via [Conda](https://github.com/conda/conda). Cocoa's internal scripts will then install any remaining missing packages, using a provided internal cache located at [cocoa_installation_libraries](https://github.com/CosmoLike/cocoa/tree/main/cocoa_installation_libraries). Assuming that the user had previously installed [Minicoda](https://docs.conda.io/en/latest/miniconda.html) (or [Anaconda](https://www.anaconda.com/products/individual)), the first step is to type the following commands to create the cocoa Conda environment.

      conda create --name cocoa python=3.7 --quiet --yes && \
      conda install -n cocoa --quiet --yes  \
      'conda-forge::libgcc-ng=10.3.0' \
      'conda-forge::libstdcxx-ng=10.3.0' \
      'conda-forge::libgfortran-ng=10.3.0' \
      'conda-forge::gxx_linux-64=10.3.0' \
      'conda-forge::gcc_linux-64=10.3.0' \
      'conda-forge::gfortran_linux-64=10.3.0' \
      'conda-forge::openmpi=4.1.1' \
      'conda-forge::sysroot_linux-64=2.17' \
      'conda-forge::git=2.33.1' \
      'conda-forge::git-lfs=3.0.2' \
      'conda-forge::hdf5=1.10.6' \
      'conda-forge::git-lfs=3.0.2' \
      'conda-forge::cmake=3.21.3' \
      'conda-forge::boost=1.76.0' \
      'conda-forge::gsl=2.7' \
      'conda-forge::fftw=3.3.10' \
      'conda-forge::cfitsio=4.0.0' \
      'conda-forge::openblas=0.3.18' \
      'conda-forge::lapack=3.9.0' \
      'conda-forge::armadillo=10.7.3'\
      'conda-forge::expat=2.4.1' \
      'conda-forge::cython=0.29.24' \
      'conda-forge::numpy=1.21.4' \
      'conda-forge::scipy=1.7.2' \
      'conda-forge::pandas=1.3.4' \
      'conda-forge::mpi4py=3.1.2' \
      'conda-forge::matplotlib=3.5.0' \
      'conda-forge::astropy=4.3.1' 
 
With this installation method, users must activate the Conda environment whenever working with Cocoa, as shown below 

    $ conda activate cocoa

**Now what? Users can now proceed to the section [Installation of Cobaya base code](#cobaya_base_code)** 

### Via Docker (best for MacOS/Windows) <a name="required_packages_docker"></a>

Docker installation will allow users to run Cocoa inside an instantiation of the [Whovian-Cosmo](https://hub.docker.com/r/vivianmiranda/whovian-cosmo) docker image. Installation of the [docker engine](https://docs.docker.com/engine/) on local PCs is a straightforward process, but it does require `sudo` privileges (see Docker's [official documentation](https://docs.docker.com/engine/install/) for OS-specific instructions).

  On macOS, type:

    $ docker run -it -p 8080:8888 -v $(pwd):/home/whovian/host/ -v ~/.ssh:/home/whovian/.ssh:ro vivianmiranda/whovian-cosmo:version-1.0.3

Linux users must type the following command instead:

    $ docker run -it -p 8080:8888 --user $(id -u):$(id -g) -v $(pwd):/home/whovian/host/ -v ~/.ssh:/home/whovian/.ssh:ro vivianmiranda/whovian-cosmo:version-1.0.3

The bind `-v ~/.ssh:/home/whovian/.ssh:ro` allows users to pull, push and clone GitHub repositories from inside the container using the host ssh keys. Users must invoke the command on the parent directory of the path where access inside the docker container is sought. 

When running the `docker run (...)/whovian-cosmo:version-1.0.3` for the first time, the docker engine will automatically download the corresponding image. This step may take some time, as the [Whovian-Cosmo](https://hub.docker.com/r/vivianmiranda/whovian-cosmo) image has approximately 700 Megabytes.

  The last step is to access the folder `/home/whovian/host/` where the host files have been mounted:

    $ cd /home/whovian/host/

**Now what? Users can now proceed to the section [Installation of Cobaya base code](#cobaya_base_code)**  

(**warning**) There isn't permanent storage outside `/home/whovian/host/`. Be aware of this fact to not lose any work

(**expert**) Most HPC systems don't allow users to run docker containers via the standard [docker engine](https://docs.docker.com/engine/) for [security reasons](https://www.reddit.com/r/docker/comments/7y2yp2/why_is_singularity_used_as_opposed_to_docker_in/?utm_source=share&utm_medium=web2x&context=3). There is, however, an alternative engine called [Singularity](https://sylabs.io/guides/3.6/user-guide/index.html) that is in compliance with most HPC requirements. The [Singularity](https://sylabs.io/guides/3.6/user-guide/index.html) engine installation requires administrative privileges, but many HPC enviroments have already adopted it. To run docker images with Singularity, go to the folder you want to store the image and type:

    $ singularity build whovian-cosmo docker://vivianmiranda/whovian-cosmo

This command will download the [Whovian-Cosmo](https://hub.docker.com/r/vivianmiranda/whovian-cosmo) image and convert it to a format that can be understood by Singularity (this might take a few minutes). To run the container interactively, type:

    $ singularity shell --no-home --bind /path/to/cocoa:/home/whovian/host --bind ~/.ssh:/home/whovian/.ssh:ro whovian-cosmo

**Now what? Users can now proceed to the section [Installation of Cobaya base code](#cobaya_base_code)**

### (expert) Via Cocoa's internal cache <a name="required_packages_cache"></a>

(**Warning**) This method is slow, not advisable. It does, however, provide the experienced user more flexibility in choosing the compiler, python and package version. Another advantage to the experienced user is that OpenMPI provided by [conda-forge](https://conda-forge.org) is [incompatible with Infiniband]((https://github.com/conda-forge/openmpi-feedstock/issues/38)). Better control may render more optimal runtime at the expense of some additional headaches! 

Whenever Conda or Docker installation procedures are unavailable, the user can still perform a local semi-autonomous installation on Linux based on a few scripts we implemented. We also provide a local copy of almost all required packages on Cocoa's cache folder named [cocoa_installation_libraries](https://github.com/CosmoLike/cocoa/tree/main/cocoa_installation_libraries) (there are HPC machines where compute nodes don't have internet access, NASA Pleiades being one example). We, therefore, only assume the pre-installation of the following packages to perform the local setup via Cocoa's internal cache:

   - [Bash](https://www.amazon.com/dp/B0043GXMSY/ref=cm_sw_em_r_mt_dp_x3UoFbDXSXRBT);
   - [Git](https://git-scm.com) v1.8+;
   - [Git LFS](https://git-lfs.github.com);
   - [gcc](https://gcc.gnu.org) v10.*;
   - [gfortran](https://gcc.gnu.org) v10.*;
   - [g++](https://gcc.gnu.org) v10.*;
   - [Python](https://www.python.org) v3.7.*;
   - [PIP package manager](https://pip.pypa.io/en/stable/installing/)
   - [Python Virtual Environment](https://www.geeksforgeeks.org/python-virtual-environment/)

To perform the local semi-autonomous installation, users should follow the procedures on section [Installation of cocoa base code](https://github.com/CosmoLike/cocoa#installation-of-cocoa-base-code), adding, however, the many additional configurations on [set_installation_options](https://github.com/CosmoLike/cocoa/blob/master/Cocoa/set_installation_options) script that are explained below.

The local installation via cocoa's internal cache is selected whenever the environmental key `MANUAL_INSTALLATION` is set:

    [Extracted from set_installation_options script] 
    
    #  ---------------------------------------------------------------------------
    # HOW COCOA BE INSTALLED? -------------------------------

    #export DOCKER_INSTALLATION=1
    #export MINICONDA_INSTALLATION=1
    #export MACOS_HOMEBREW_INSTALLATION=1
    export MANUAL_INSTALLATION=1
    
The user also needs to set the following self-explanatory environmental keys on [set_installation_options](https://github.com/CosmoLike/cocoa/blob/master/Cocoa/set_installation_options):
 
    [Extracted from set_installation_options script]
  
    elif [ -n "${MANUAL_INSTALLATION}" ]; then

      export GLOBAL_PACKAGES_LOCATION=/usr/local
      export PYTHON_VERSION=3
      export FORTRAN_COMPILER=gfortran
    
      export C_COMPILER=gcc
      export CXX_COMPILER=g++
      export GLOBALPYTHON3=python3
      export MPI_FORTRAN_COMPILER=mpif90
      export MPI_CXX_COMPILER=mpicc
      export MPI_CC_COMPILER=mpicxx
    
      # In case global packages are available 
      #export IGNORE_DISTUTILS_INSTALLATION=1
      #export IGNORE_OPENBLAS_INSTALLATION=1
      #export IGNORE_XZ_INSTALLATION=1
      #export IGNORE_ALL_PIP_INSTALLATION=1
      #export IGNORE_CMAKE_INSTALLATION=1
      #export IGNORE_CPP_BOOST_INSTALLATION=1
      #export IGNORE_CPP_ARMA_INSTALLATION=1
      #export IGNORE_CPP_SPDLOG_INSTALLATION=1
      #export IGNORE_C_GSL_INSTALLATION=1
      #export IGNORE_C_CFITSIO_INSTALLATION=1
      #export IGNORE_C_FFTW_INSTALLATION=1 
      #export IGNORE_OPENBLAS_INSTALLATION=1
      #export IGNORE_FORTRAN_LAPACK_INSTALLATION=1
   
(**expert**) Our scripts never install packages on `$HOME/.local`. Doing so could impose incompatibilities between Cobaya and different projects (or break the user's environment for other projects). All requirements for Cocoa are installed at

    Cocoa/.local/bin
    Cocoa/.local/include
    Cocoa/.local/lib
    Cocoa/.local/share

**Now what? Users can now proceed to the section [Installation of Cobaya base code](#cobaya_base_code)**

## Installation of Cobaya base code <a name="cobaya_base_code"></a>

Type:

    $ git lfs clone git@github.com:CosmoLike/cocoa.git

to clone the repository. 

(**Warning**) We have a limited monthly quota in bandwidth for [Git LFS](https://git-lfs.github.com) files, and therefore we ask users to use good judgment in the number of times they clone repositories with large files (each clone will download around 5GB from Git LFS).

Cocoa is made aware of the chosen installation method of required packages via special environment keys located on the [set_installation_options](https://github.com/CosmoLike/cocoa/blob/master/Cocoa/set_installation_options) script, as shown below

    [Extracted from set_installation_options script]
    #  ---------------------------------------------------------------------------
    # HOW COCOA BE INSTALLED? -------------------------------

    #export DOCKER_INSTALLATION=1
    #export MINICONDA_INSTALLATION=1
    #export MACOS_HOMEBREW_INSTALLATION=1
    #export MANUAL_INSTALLATION=1
    
The user must uncomment the appropriate key, and then type the following command

    source setup_cocoa_installation_packages

(**expert**) This script decompress the data files and install all packages that may have been left out in the Conda/Docker/Homebrew/Manual installation. File decompression should only take a few minutes, while package installation time may range from a few minutes (if installation via *Conda*, *Docker* or *Homebrew* was selected) to more than one hour (if installation *via Cocoa's internal scripts and cache* was selected). 

Finally, type

    source compile_external_modules
    
to compile CAMB, CLASS, Planck and Polychord.

(**expert**) If the user wants recompile just one of these packages, read the appendix [Compiling Boltzmann, CosmoLike and Likelihood codes separatelly](#compile_separatelly).

## Running Cobaya Examples <a name="cobaya_base_code_examples"></a>

Assuming the user opted for the easier *Conda installation* and located the terminal at the folder *where Cocoa was cloned*, this is how to run some example YAML files we provide (*no Cosmolike code involved*): 

**Step 1 of 5**: activate the conda environment

    $ conda activate cocoa
     
**Step 2 of 5**: go to the Cocoa main folder 

    $(cocoa) cd ./cocoa/Cocoa

(**warning**) Always run our provided scripts from the Cocoa main folder

**Step 3 of 5**: activate the private python environment

    $(cocoa) source start_cocoa

(**expert**) Users will see a terminal that looks like this: `$(Cocoa)(.local)`. *This is a feature, not a bug*! The Conda environment can be the same for all Cocoa instances, with [start_cocoa](https://github.com/CosmoLike/cocoa/blob/master/Cocoa/start_cocoa)/[stop_cocoa](https://github.com/CosmoLike/cocoa/blob/master/Cocoa/stop_cocoa) loading/unloading the corresponding `LD_LIBRARY_PATH`, `CPATH`, `C_INCLUDE_PATH`, `CPLUS_INCLUDE_PATH` and `PATH`. *Why more than one Cocoa instance?* While users may be running chains in one instance, they might use a second instantiation to make experimental changes.

**Step 4 of 5**: perform a basic OpenMP setup
    
    $(cocoa)(.local) export OMP_NUM_THREADS = [1-4]

**Step 5 of 5**: run `cobaya-run` on a template YAML files we provide

     $(cocoa)(.local) mpirun -n 1 --mca btl tcp,self --bind-to core --rank-by core --map-by numa:pe=${OMP_NUM_THREADS} cobaya-run ./projects/example/EXAMPLE_EVALUATE[1-4].yaml -f
     
or  

     $(cocoa)(.local) mpirun -n 4 --mca btl tcp,self --bind-to core --rank-by core --map-by numa:pe=${OMP_NUM_THREADS} cobaya-run ./projects/example/EXAMPLE_MCMC[1-3].yaml -f

(**expert**) Why the `--mca btl tcp,self` flag? Conda-forge developers don't [compile OpenMPI with Infiniband compatibility](https://github.com/conda-forge/openmpi-feedstock/issues/38). Users outraged by the overhead TCP will bring over Infiniband can perform the [installation via Cocoa's internal cache](required_packages_cache). 

(**expert**) Why the `--bind-to core --rank-by core --map-by numa:pe=${OMP_NUM_THREADS}` flag? To enable hybrid MPI + OpenMP run at UofA's supercomputer. *Users should check if the flag is necessary on their particular environment.*

Once the work is done, type:

    $(cocoa)(.local) source stop_cocoa

(**expert**) [stop_cocoa](https://github.com/CosmoLike/cocoa/blob/master/Cocoa/stop_cocoa) will also restore `OMP_NUM_THREADS` to its original value before the script [start_cocoa](https://github.com/CosmoLike/cocoa/blob/master/Cocoa/start_cocoa) was sourced. 

and (optional)  
    
    $(cocoa) conda deactivate cocoa

# Running Cosmolike projects <a name="running_cosmolike_projects"></a> 

The projects folder was designed to include all Cosmolike projects. Similar to last section, we assume the user opted for the easier *Conda installation*, and located the terminal at the folder *where Cocoa was cloned*.

**Step 1 of 4**: go to the project folder (`./cocoa/Cocoa/projects`) and clone a cosmolike project, as shown below:

    $ cd ./cocoa/Cocoa/projects
    $ git clone git@github.com:CosmoLike/cocoa_XXX.git XXX
    $ cd ../

(**warning**) The Cosmolike Organization hosts a Cobaya-Cosmolike project named XXX at `git@github.com:CosmoLike/cocoa_XXX.git`. The prefix `cocoa_` avoids mixing Cobaya-Cosmolike projects with code meant to be run on the legacy CosmoLike code. However, our provided scripts and template YAML files assume the removal of `cocoa_` prefix. Indeed, the command `git clone git@github.com:CosmoLike/cocoa_XXX.git XXX` clone the repository *and* remove the prefix `cocoa_`.

**Step 2 of 4**: activate conda Cocoa environment and the private python environment

     $ conda activate cocoa
     $(cocoa) source start_cocoa
 
 (**warning**): Please run the `start_cocoa` *after* cloning the cosmolike repository. 
 
 **Step 3 of 4**: compile the project
 
     $(cocoa)(.local) source ./projects/XXX/scripts/compile_XXX
  
 **Step 4 of 4**: perform a basic OpenMP setup and run a template yaml file we provide
    
    $(cocoa)(.local) export OMP_NUM_THREADS = [1-4]
    $(cocoa)(.local) mpirun -n 1 --mca btl tcp,self --bind-to core --rank-by core --map-by numa:pe=${OMP_NUM_THREADS} cobaya-run ./projects/XXX/EXAMPLE_EVALUATE1.yaml -f

# Creating Cosmolike projects <a name="creating_cosmolike_projects"></a> 

The `XXX` project needs to have the more or less the following structure (taken from our private DES-Y3 project)

    +-- cocoa_des_y3
    |    +-- likelihood
    |    |   +-- _cosmolike_prototype_base.py
    |    |   +-- des_3x2pt.py
    |    |   +-- des_3x2pt.yaml
    |    |   +-- des_2x2pt.py
    |    |   +-- des_3x2pt.yaml
    |    |   +-- des_cosmic_shear.py
    |    |   +-- des_cosmic_shear.yaml
    |    +-- scripts
    |    |   +-- compile_des_y3
    |    |   +-- start_des_y3
    |    |   +-- stop_des_y3
    |    +-- data
    |    |   +-- DES.paramnames
    |    |   +-- DES_Y3.dataset
    |    |   +-- datavector.txt
    |    |   +-- covariance.txt
    |    |   +-- nzlens.txt
    |    |   +-- nzsource.txt
    |    |   +-- mask.mask
    |    +-- interface
    |    |   +-- MakefileCosmolike
    |    |   +-- cosmolike_des_y3_interface.py
    |    |   +-- interface.cpp
    |    |   +-- interface.hpp
    |    +-- chains
    |    |   +-- README

We expect to have a public project that can be used as a template for new Cosmolike projects in the near future.

## Appendix <a name="appendix"></a>

### Proper Credits <a name="appendix_proper_credits"></a>

The following is not an exhaustive list of the codes we use

- [Cobaya](https://github.com/CobayaSampler) is a framework developed by Dr. Jesus Torrado and Prof. Anthony Lewis

- [Cosmolike](https://github.com/CosmoLike) is a framework developed by Prof. Elisabeth Krause and Prof. Tim Eifler

- [CAMB](https://github.com/cmbant/CAMB) is a Boltzmann code developed by Prof. Anthony Lewis

- [CLASS](https://github.com/lesgourg/class_public) is a Boltzmann code developed by Prof. Julien Lesgourgues and Dr. Thomas Tram

- [Polychord](https://github.com/PolyChord/PolyChordLite) is a sampler code developed by Dr. Will Handley, Prof. Lasenby, and Prof. M. Hobson

By no means, we want to discourage people from cloning code from their original repositories. We've included these codes as compressed [xz file format](https://tukaani.org/xz/format.html) in our repository for convenience in the initial development. The work of those authors is extraordinary, and they must be properly cited.

### Compiling Boltzmann, CosmoLike and Likelihood codes separatelly <a name="appendix_compile_separatelly"></a>

To avoid excessive compilation times during development, users can use specialized scripts that compile only the specific modules:

    $(.local) source ./installation_scripts/setup_class

    $(.local) source ./installation_scripts/setup_camb

    $(.local) source ./installation_scripts/setup_planck

    $(.local) source ./installation_scripts/setup_polychord

### Running Jupyter Notebooks inside the [Whovian-Cosmo](https://hub.docker.com/r/vivianmiranda/whovian-cosmo) docker container <a name="appendix_jupyter_whovian"></a>

[Cobaya](https://github.com/CobayaSampler), the framework that Cocoa heavily depends on has excellent integration with Jupyter notebooks. Below, some in-depth instructions to run notebooks inside the [Whovian-Cosmo](https://hub.docker.com/r/vivianmiranda/whovian-cosmo) docker container

To start a jupyter notebook, type the following command on the docker container:

    $ jupyter notebook --no-browser

The terminal will show a message similar to the following template:

    [... NotebookApp] Writing notebook server cookie secret to /home/whovian/.local/share/jupyter/runtime/notebook_cookie_secret
    [... NotebookApp] WARNING: The notebook server is listening on all IP addresses and not using encryption. This is not recommended.
    [... NotebookApp] Serving notebooks from local directory: /home/whovian/host
    [... NotebookApp] Jupyter Notebook 6.1.1 is running at:
    [... NotebookApp] http://f0a13949f6b5:8888/?token=XXX
    [... NotebookApp] or http://127.0.0.1:8888/?token=XXX
    [... NotebookApp] Use Control-C to stop this server and shut down all kernels (twice to skip confirmation).

where `XXX` in the line `[... NotebookApp] or http://127.0.0.1:8888/?token=XXX` is the token you need to save to access the notebook. We will assume you are running the docker container in a host server with URL `your_sever.com` that you are accessing via ssh. From your local PC type:

    $ ssh your_username@your_sever.com -L 8080:localhost:8080

   Finally, go to your browser and type `http://localhost:8080/?token=XXX`, where `XXX` is the previously saved token. For security, we do not allow password-based connections to the jupyter notebooks.

### Summary Information about Cocoa's configuration files <a name="appendix_config_files"></a>

The installation of Cocoa required packages, as well as Boltzmann and Likelihood codes, are managed via the following scripts located at `./Cocoa`.

 - [set_installation_options](https://github.com/CosmoLike/cocoa/blob/master/Cocoa/set_installation_options)

    This file contains environment variables that manage the installation process.

 - [setup_cocoa_installation_packages](https://github.com/CosmoLike/cocoa/blob/master/Cocoa/setup_cocoa_installation_packages)

    This file has instructions on how to install packages required by the Cocoa Framework.

 - [compile_external_modules](https://github.com/CosmoLike/cocoa/blob/master/Cocoa/compile_external_modules)

    This file has instructions on how to compile Boltzmann, Sampler and likelihood codes. 

 - [start_cocoa](https://github.com/CosmoLike/cocoa/blob/master/Cocoa/start_cocoa)

    This file has instructions on how to set up the Python virtual environment.

 - [stop_cocoa](https://github.com/CosmoLike/cocoa/blob/master/Cocoa/stop_cocoa)

    This file has instructions on how to unset the Python virtual environment - including recovering original `PYTHONPATH`, `LD_LIBRARY_PATH`, and `PATH`. 

 - [clean_all](https://github.com/CosmoLike/cocoa/blob/master/Cocoa/clean_all)

    This file has instructions on how to clean keys associated with the Python virtual environment and delete the compilation of the Boltzmann, Sampler, and likelihood codes, and local installation of the required packages installed by the [setup_cocoa_installation_packages].
