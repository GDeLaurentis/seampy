import numpy
import pytest

import lips
import seampy

from antares.core.bh_unknown import BHUnknown
from antares.core.se_unknown import SEUnknown


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #


@pytest.mark.parametrize(
    "helconf", ["ppm", "ppmm", "pmpm", "pppmm", "ppmpm", "pmpmpm", "ppmpmm", "pppmmm", "pppmmmm", "pppppmm", "ppmpmmm", "pmpmpmm"]
)
def test_YM_with_BlackHat(helconf):
    oParticles = lips.Particles(len(helconf))
    oParticles.fix_mom_cons(real_momenta=False)
    oBHUnknown = BHUnknown(helconf=helconf, amppart="tree")
    oCHYUnknown = seampy.NumericalAmplitude(theory="YM", helconf=helconf)
    assert (numpy.isclose(complex(oBHUnknown(oParticles)), complex(oCHYUnknown(oParticles))) or
            numpy.isclose(complex(oBHUnknown(oParticles)), -complex(oCHYUnknown(oParticles))))


@pytest.mark.parametrize(
    ("helconf, se_arg"),
    [("ppmm", "\"YM\", 0, m, \"g+\", \"g+\", \"g-\", \"g-\""),
     ("pmpm", "\"YM\", 0, m, \"g+\", \"g-\", \"g+\", \"g-\""),
     ("pppmm", "\"YM\", 0, m, \"g+\", \"g+\", \"g+\", \"g-\", \"g-\""),
     ("ppmpm", "\"YM\", 0, m, \"g+\", \"g+\", \"g-\", \"g+\", \"g-\""),
     ("pppmmm", "\"YM\", 0, m, \"g+\", \"g+\", \"g+\", \"g-\", \"g-\", \"g-\""),
     ("ppmpmm", "\"YM\", 0, m, \"g+\", \"g+\", \"g-\", \"g+\", \"g-\", \"g-\"")]
)
def test_YM_with_JoeSEsolver(helconf, se_arg):
    oParticles = lips.Particles(len(helconf))
    oParticles.fix_mom_cons(real_momenta=False)
    oSEUnknown = SEUnknown(se_arg)
    oCHYUnknown = seampy.NumericalAmplitude(theory="YM", helconf=helconf)
    assert (numpy.isclose(1j * complex(oSEUnknown(oParticles)), complex(oCHYUnknown(oParticles))) or
            numpy.isclose(1j * complex(oSEUnknown(oParticles)), -complex(oCHYUnknown(oParticles))))


@pytest.mark.parametrize(
    ("helconf, se_arg"),
    [("ppmm", "\"EG\", 0, m, \"h+\", \"h+\", \"h-\", \"h-\""),
     ("pppmm", "\"EG\", 0, m, \"h+\", \"h+\", \"h+\", \"h-\", \"h-\""),
     ("ppppmm", "\"EG\", 0, m, \"h+\", \"h+\", \"h+\", \"h+\", \"h-\", \"h-\""),
     ("pppmmm", "\"EG\", 0, m, \"h+\", \"h+\", \"h+\", \"h-\", \"h-\", \"h-\"")]
)
def test_EG_with_JoeSEsolver(helconf, se_arg):
    oParticles = lips.Particles(len(helconf))
    oParticles.fix_mom_cons(real_momenta=False)
    oSEUnknown = SEUnknown(se_arg)
    oCHYUnknown = seampy.NumericalAmplitude(theory="EG", helconf=helconf)
    assert (numpy.isclose(1j * complex(oSEUnknown(oParticles)), complex(oCHYUnknown(oParticles))) or
            numpy.isclose(1j * complex(oSEUnknown(oParticles)), -complex(oCHYUnknown(oParticles))))


@pytest.mark.parametrize(
    ("helconf, se_arg"),
    [("ppmm", "\"CG\", 4, m, \"h+\", \"h+\", \"h-\", \"h-\""),
     ("pppmm", "\"CG\", 4, m, \"h+\", \"h+\", \"h+\", \"h-\", \"h-\""),
     ("ppppmm", "\"CG\", 4, m, \"h+\", \"h+\", \"h+\", \"h+\", \"h-\", \"h-\""),
     ("pppmmm", "\"CG\", 4, m, \"h+\", \"h+\", \"h+\", \"h-\", \"h-\", \"h-\"")]
)
def test_CG_with_JoeSEsolver(helconf, se_arg):
    oParticles = lips.Particles(len(helconf))
    oParticles.fix_mom_cons(real_momenta=False)
    oSEUnknown = SEUnknown(se_arg)
    oCHYUnknown = seampy.NumericalAmplitude(theory="CG", helconf=helconf)
    assert (numpy.isclose(1j * complex(oSEUnknown(oParticles)), complex(oCHYUnknown(oParticles))) or
            numpy.isclose(1j * complex(oSEUnknown(oParticles)), -complex(oCHYUnknown(oParticles))))
