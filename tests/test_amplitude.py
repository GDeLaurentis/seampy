import numpy
import pytest

import lips
import seampy

from antares.core.unknown import BHUnknown

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #


def test_CG_not_color_ordered():
    oParticles = lips.Particles(6)
    oParticles.fix_mom_cons()
    oNewParticles = oParticles.image("321456")
    oCGAmp = seampy.NumericalAmplitude(theory="CG", helconf="pmpmpm")
    assert(abs(oCGAmp(oParticles) - oCGAmp(oNewParticles)) < 10 ** -270)


@pytest.mark.parametrize(
    "helconf", ["ppm", "ppmm", "pmpm", "pppmm", "ppmpm", "pmpmpm", "ppmpmm", "pppmmm", "pppmmmm"]
)
def test_YM(helconf):
    oParticles = lips.Particles(len(helconf))
    oParticles.fix_mom_cons(real_momenta=False)
    oBHUnknown = BHUnknown(helconf=helconf, amppart="tree")
    oCHYUnknown = seampy.NumericalAmplitude(theory="YM", helconf=helconf)
    assert(numpy.isclose(complex(oBHUnknown(oParticles)), complex(oCHYUnknown(oParticles))) or
           numpy.isclose(complex(oBHUnknown(oParticles)), -complex(oCHYUnknown(oParticles))))
