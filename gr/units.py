"""
    units.py

    Defines common systems of units and conversions between them.

    Units included are:

             System     | Class |    Notes
        ----------------|-------|--------------
        Relativistic    |  REL  | G = c = M = 1
        Gaussian        |  CGS  |
        SI (MKS)        |  MKS  |

    All values for units listed below are from http://pdg.lbl.gov.
"""


class SystemOfUnits:
    """Defines a system of units."""

    def __init__(self, G, M, c):
        """Setup a system of units based on the values of the gravitational 
        constant, solar mass, and speed of light.

        Parameters
        ----------
        G : float
            Newton's gravitational constant
        M : float
            Solar mass
        c : float
            Speed of light
        """
        self.G = G
        self.M = M
        self.c = c

        # For more verbose reference
        self.mass = M

        # Define commonly used dimensions
        self.length = M*G / (c**2)
        self.time = self.length / c
        self.density = M / (self.length**3)
        self.force = M * self.length / (self.time**2)
        self.pressure = self.force / (self.length**2)
        self.energy = self.force * self.length
        self.energy_density = self.energy / (self.length**3)
        self.specific_energy = self.energy / (self.mass)


### Some commonly used systems ###

# Relativistic units
REL = SystemOfUnits(G=1.0,  # Gravitational constant
                    M=1.0,  # Solar mass
                    c=1.0)  # Speed of light

# Gaussian units
CGS = SystemOfUnits(G=6.67408e-8,     # Gravitational constant [cm^3 g^-1 s^-2]
                    M=1.98848e33,     # Solar mass [g]
                    c=29979245800.0)  # Speed of light [cm s^-1]

# SI units
MKS = SystemOfUnits(G=6.67408e-11,  # Gravitational constant [m^3 kg^-1 s-2]
                    M=1.98848e30,   # Solar mass [kg]
                    c=299792458.0)  # Speed of light [m s^-1]


def convert(quantity, old_units, new_units):
    """Convert a quantity from one system of units to another.

    Note
    ----
    The `old_units` and `new_units` should be attributes from a defined system 
    of units, e.g. to convert a CGS density to MKS:

        `new_rho = convert(old_rho, CGS.density, MKS.density)`

    Parameters
    ----------
    quantity : float (or array of floats)
        The quantity to convert
    old_units : float
        The units being converted from.
    new_units : float
        The units being converted to

    Returns
    -------
    float
        The converted quantity
    """

    return quantity * new_units / old_units
