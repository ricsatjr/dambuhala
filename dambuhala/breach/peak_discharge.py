def wang_etal_2018_eq4(Vw,Hw,g=9.81):
  """
  Vw   volume of reservoir water above the breach, in m^3
  Hw   height of reservoir water above the breach, in m
  g    acceleration due to gravity, in m/s2

  Return
  Qp   peak discharge, in m^3/s

  from Bo Wang, Yunliang Chen, Chao Wu, Yong Peng, Jiajun Song, Wenjun Liu, Xin Liu.
  Empirical and semi-analytical models for predicting peak outflows caused by embankment dam failures,
  Journal of Hydrology,
  Volume 562,
  2018,
  Pages 692-702,
  https://doi.org/10.1016/j.jhydrol.2018.05.049
  """
  M_Vw=Vw/(1e6)  # convert to million m^3
  Qp = 0.0370 *  ((M_Vw/(Hw**3))**-0.4262)  *  (g*(M_Vw**(5/3)))*0.5
  return Qp