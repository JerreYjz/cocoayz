#include <stdio.h>
#include <gsl/gsl_math.h>
#include <gsl/gsl_sf.h>

#include "basics.h"
#include "cosmo3D.h"
#include "structs.h"

#include "log.c/src/log.h"

//{ Phistat, alpha, Mstar, Q, P} [0][] = red, [1][] = all
static double LF_coefficients[2][5] =
 {
    {-1.,0.,0.,0.,0.},
    {-1.,0.,0.,0.,0.}
  };

static double LF_coefficients_GAMA[2][5] =
  {
    {1.11e-3,-0.57,-20.34,1.8,-1.2},
    {0.94e-3,-1.23,-20.70,0.7,1.8}
  };//from GAMA survey http://arxiv.org/pdf/1111.0166v2.pdf

static double LF_coefficients_DEEP2[2][5] =
  {
    {1.11e-3,-0.57,-20.34,1.20,-1.15},
    {0.94e-3,-1.23,-20.70,1.23,-.3}
  }; //P,Q from DEEP2 (B-band), otherwise GAMA

// ---------------------------------------------------------------------------
// k+e corrections for early types, restframe r band at
// z = 0.,0.1,..,3.0; interpolated from
// http://vizier.u-strasbg.fr/viz-bin/VizieR?-source=J/A%2BAS/122/399
// ---------------------------------------------------------------------------
static double KE[31] = {
  0. ,   -0.018,  0.013,  0.043,  0.091,  0.236,  0.449,
  0.667,  0.827,  0.907,  0.916,  0.91 ,  0.932,  0.901,
  0.835,  0.735,  0.594,  0.427,  0.179, -0.025, -0.226,
 -0.423, -0.591, -0.754, -0.913, -1.061, -1.181, -1.301,
 -1.421, -1.541, -1.661
};

void set_LF_GAMA(void)
{
  for (int i = 0; i <5; i++)
  {
    LF_coefficients[0][i] =LF_coefficients_GAMA[0][i];
    LF_coefficients[1][i] =LF_coefficients_GAMA[1][i];
  }
}

void set_LF_DEEP2(void)
{
  for (int i = 0; i <5; i++)
  {
    LF_coefficients[0][i] = LF_coefficients_DEEP2[0][i];
    LF_coefficients[1][i] = LF_coefficients_DEEP2[1][i];
  }
}

double M_abs(double mag, double a)
{ //in h = 1 units, incl. Poggianti 1997 k+e-corrections
  static double* table;

  if (table == 0)
  {
    // read in + tabulate k+e corrections for early types, restframe r band
    // interpolated from http://vizier.u-strasbg.fr/viz-bin/VizieR?-source=J/A%2BAS/122/399
    const int size = 31;
    table = (double*) malloc(sizeof(double)*size);
    for (int i = 0; i<size; i++)
    {
      table[i] = KE[i];
    }
  }

  // no acceptable k-korrection exists for k>3, also no meaningful IA model
  const double z = 1./a - 1.0 >= 3.0 ? 2.99 : 1./a - 1.0;
  const double ke = interpol(table, 31, 0., 3.0, 0.1, z, 1.0, 1.0);
  struct chis chidchi = chi_all(a);
  const double fK = f_K(chidchi.chi);

  return mag - 5.0*log10(fK/a*cosmology.coverH0) - 25.0 - ke;
}

double f_red_LF(double mag, double a)
{
  if (LF_coefficients[0][0]< 0)
  {
    log_fatal("Missing Luminosity function");
    exit(1);
  }

  double LF_all[3], LF_red[3];

  // r-band LF parameters, Tab. 5 in http://arxiv.org/pdf/1111.0166v2.pdf

  //red galaxies
  const double alpha_red = LF_coefficients[0][1] + nuisance.LF_red_alpha;
  const double Mstar_red = LF_coefficients[0][2]; //in h = 1 units
  const double Q_red = LF_coefficients[0][3] + nuisance.LF_red_Q;
  const double P_red = LF_coefficients[0][4] + nuisance.LF_red_P;
  const double Phistar_red = LF_coefficients[0][0];

  LF_red[0] = Phistar_red*pow(10.0,0.4*P_red*(1./a-1));
  LF_red[1] = Mstar_red-Q_red*(1./a-1. -0.1);
  LF_red[2]= alpha_red;

  //all galaxies
  const double alpha =LF_coefficients[1][1] + nuisance.LF_alpha;
  const double Mstar = LF_coefficients[1][2];
  const double Q = LF_coefficients[1][3] + nuisance.LF_Q;
  const double P = LF_coefficients[1][4] + nuisance.LF_P;
  const double Phistar = LF_coefficients[1][0];

  LF_all[0] = Phistar*pow(10.0,0.4*P*(1./a-1));
  LF_all[1] = Mstar -Q*(1./a-1. - 0.1);
  LF_all[2] = alpha;

  const double Mlim = M_abs(mag,a); //also in h = 1 units

  return LF_red[0]/LF_all[0]*
    gsl_sf_gamma_inc(LF_red[2]+1, pow(10.0,-0.4*(Mlim-LF_red[1])))/
    gsl_sf_gamma_inc(LF_all[2]+1, pow(10.0,-0.4*(Mlim-LF_all[1])));
}

// averaged (L/L_0)^beta over red galaxy LF
double A_LF(double mag, double a)
{
  if (LF_coefficients[0][0]< 0)
  {
    log_fatal("Missing Luminosity function");
    exit(1);
  }

  // r-band LF parameters, Tab. 5 in http://arxiv.org/pdf/1111.0166v2.pdf

  //red galaxies
  const double alpha_red = LF_coefficients[0][1] + nuisance.LF_red_alpha;
  const double Mstar_red = LF_coefficients[0][2]; //in h = 1 units
  const double Q_red = LF_coefficients[0][3] + nuisance.LF_red_Q;

  double LF_red[3];
  LF_red[1] = Mstar_red-Q_red*(1./a-1. -.1);
  LF_red[2]= alpha_red;

  const double Mlim = M_abs(mag,a);
  const double Lstar = pow(10.0, -0.4*LF_red[1]);
  const double x = pow(10.0, -0.4*(Mlim - LF_red[1])); //Llim/Lstar
  const double L0 =pow(10.0, -0.4*(-22.)); //all in h = 1 units

  return pow(Lstar/L0, nuisance.beta_ia)*
    gsl_sf_gamma_inc(LF_red[2] + nuisance.beta_ia+1, x)/
    gsl_sf_gamma_inc(LF_red[2] + 1, x);
}

// ---------------------------------------------------------------------------
// return 1 if combination of all + red galaxy LF parameters is unphysical,
// i.e. if f_red > 1 for some z < redshift.shear_zdistrpar_zmax
// ---------------------------------------------------------------------------
int check_LF(void)
{
  double a = 1./(1+redshift.shear_zdistrpar_zmax) + 0.005;
  while (a < 1.)
  {
    if (M_abs(survey.m_lim,a) < LF_coefficients[1][2] - (LF_coefficients[1][3]
                                  + nuisance.LF_Q)*(1./a-1. - 0.1) ||
        M_abs(survey.m_lim,a) < LF_coefficients[0][2] - (LF_coefficients[0][3]
                                  + nuisance.LF_red_Q)*(1./a-1. - 0.1))
    {
      return 1;
    }
    if (f_red_LF(survey.m_lim,a) > 1.0)
    {
      return 1;
    }

    a += 0.01;
  }
  return 0;
}

double A_IA_Joachimi(const double a)
{
  const double highz = 0.75;
  const double z = 1./a - 1;

  // A_0* < (L/L_0)^beta > *f_red
  const double A_red = nuisance.A_ia*A_LF(survey.m_lim,a)*f_red_LF(survey.m_lim,a);

  if (a < 1./(1.+ highz))
  { // z > highz, factor in uncertainty in extrapolation of redshift scaling
    return A_red*
      pow((1.0 + z)/nuisance.oneplusz0_ia, nuisance.eta_ia)*
      pow((1.0 + z)/(1. + highz), nuisance.eta_ia_highz);
  }
  else
  { //standard redshift scaling
    return A_red*pow((1. + z)/nuisance.oneplusz0_ia, nuisance.eta_ia);
  }
}

void IA_A1_Z1Z2(const double a, const double growfac_a, const int n1, const int n2, double res[2])
{ 
  // COCOA (WARNING): THERE IS MINUS SIGN DIFFERENCE COMPARED TO C1_TA 
  // COCOA (WARNING): IN ORIGINAL COSMOLIKE (SEE: cosmo2D_fullsky_TATT.c)

  if (!(a>0)) 
  {
    log_fatal("a>0 not true");
    exit(1);
  }

  const double norm = cosmology.Omega_m*nuisance.c1rhocrit_ia/growfac_a;
  double A_Z1 = 0.0;
  double A_Z2 = 0.0;
  
  const int IA = abs(like.IA);

  switch(IA)
  {
    case NO_IA:
    {
      A_Z1 = 0.0;
      A_Z2 = 0.0;
      break;
    }
    case IA_NLA_LF:
    {
      A_Z1 = A_IA_Joachimi(a);
      A_Z2 = A_Z1;
      break;
    }
    case IA_REDSHIFT_BINNING:
    { 
      A_Z1 = nuisance.A_z[n1];
      A_Z2 = nuisance.A_z[n2];
      break;
    }
    case IA_REDSHIFT_EVOLUTION:
    {
      const double oneplusz = (1.0/a);
      const double x = oneplusz/nuisance.oneplusz0_ia;
      A_Z1 = nuisance.A_ia*pow(x, nuisance.eta_ia);
      A_Z2 = A_Z1;
      break;
    }
    default:
    {
      log_fatal("like.IA = %d not supported", like.IA);
      exit(1);
    }
  }
  
  res[0] = A_Z1 * norm;
  res[1] = A_Z2 * norm;

  return;
}

double IA_A1_Z1(const double a, const double growfac_a, const int n1)
{
  double res[2];
  IA_A1_Z1Z2(a, growfac_a, n1, n1, res);
  return res[0];
}

void IA_A2_Z1Z2(const double a, const double growfac_a, 
  const int n1, const int n2, double res[2])
{
  // COCOA (WARNING): THERE IS factor x5 DIFFERENCE COMPARED TO C2_TT 
  // COCOA (WARNING): IN ORIGINAL COSMOLIKE (SEE: cosmo2D_fullsky_TATT.c)

  if (!(a>0)) 
  {
    log_fatal("a>0 not true");
    exit(1);
  }

  const double norm_a2 = 
    cosmology.Omega_m*nuisance.c1rhocrit_ia/(growfac_a*growfac_a);
  double A2_Z1 = 0.0;
  double A2_Z2 = 0.0;

  const int IA = abs(like.IA);

  switch(IA)
  {
    case NO_IA:
    {
      A2_Z1 = 0.0;
      A2_Z2 = 0.0;
      break;
    }
    case IA_NLA_LF:
    {
      log_fatal("IA_NLA_LF TT not supported");
      exit(1);      
    }
    case IA_REDSHIFT_BINNING:
    { 
      A2_Z1 = nuisance.A2_z[n1];
      A2_Z2 = nuisance.A2_z[n2];
      break;
    }
    default:
    {
      const double oneplusz = (1.0/a);
      const double x = oneplusz/nuisance.oneplusz0_ia;
      A2_Z1 = nuisance.A2_ia*pow(x, nuisance.eta_ia_tt);
      A2_Z2 = A2_Z1;
    } 
  }

  res[0] = A2_Z1 * norm_a2;
  res[1] = A2_Z2 * norm_a2;

  return;
}

double IA_A2_Z1(const double a, const double growfac_a, const int n1)
{
  double res[2];
  IA_A2_Z1Z2(a, growfac_a, n1, n1, res);
  return res[0];
}

void IA_BTA_Z1Z2(const double a __attribute__((unused)), 
  const double growfac_a __attribute__((unused)), 
  const int n1, const int n2, double res[2])
{
  double BTA_Z1 = 0.0;
  double BTA_Z2 = 0.0;

//  if (!(a>0)) 
//  {
//    log_fatal("a>0 not true");
//    exit(1);
//  }

  const int IA = abs(like.IA);

  switch(IA)
  {
    case NO_IA:
    {
      BTA_Z1 = 1.0;
      BTA_Z2 = 1.0;
      break;
    }
    case IA_NLA_LF:
    {
      log_fatal("IA_NLA_LF BTA not supported");
      exit(1);      
    }
    case IA_REDSHIFT_BINNING:
    {
      BTA_Z1 = nuisance.b_ta_z[n1];
      BTA_Z2 = nuisance.b_ta_z[n2];
      break;
    }
    default:
    {
      BTA_Z1 = nuisance.b_ta_z[0];
      BTA_Z2 = BTA_Z1;
    }
  }

  res[0] = BTA_Z1;
  res[1] = BTA_Z2;

  return;
}

double IA_BTA_Z1(const double a, const double growfac_a, const int n1)
{
  double res[2];
  IA_BTA_Z1Z2(a, growfac_a, n1, n1, res);
  return res[0];
}