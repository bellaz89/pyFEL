'''
    Physical constants
'''
from scipy.constants import c, e, epsilon_0, physical_constants

C = c
ELEM_CHARGE = e
VACUUM_IMPEDANCE = 1.0/(c*epsilon_0)
ELECTRON_MASS_EV = physical_constants['electron mass energy equivalent in MeV'][0]*1e6

if __name__ == "__main__":
    print("Speed of light                       (m/s):", C)
    print("Elementary charge                      (C):", ELEM_CHARGE)
    print("Speed of light x elementary charge (C*m/s):", C*ELEM_CHARGE)
    print("Vacuum impedance                     (ohm):", VACUUM_IMPEDANCE)
    print("Electron mass                         (eV):", ELECTRON_MASS_EV)

