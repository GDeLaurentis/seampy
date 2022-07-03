import numpy
import pytest

import lips
import seampy

from antares.core.bh_unknown import BHUnknown
from antares.core.se_unknown import SEUnknown


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #


def test_CG_not_color_ordered():
    oParticles = lips.Particles(6)
    oNewParticles = oParticles.image("321456")
    oCGAmp = seampy.NumericalAmplitude(theory="CG", helconf="pmpmpm")
    assert(abs(oCGAmp(oParticles) - oCGAmp(oNewParticles)) < 10 ** -270)


@pytest.mark.parametrize(
    ("helconf", "analytic_expr"),
    [("ppm", "([12]⁴)/([12][23][31])"),
     ("ppmm", "([12]⁴)/([12][23][34][41])"),
     ("pmpm", "([13]⁴)/([12][23][34][41])"),
     ("pppmm", "(⟨45⟩⁴)/(⟨12⟩⟨23⟩⟨34⟩⟨45⟩⟨51⟩)"),
     ("ppmpm", "(⟨35⟩⁴)/(⟨12⟩⟨23⟩⟨34⟩⟨45⟩⟨51⟩)"),
     ("pppmmm", """+1⟨4|(2+3)|1]³/([1|6]⟨2|3⟩⟨3|4⟩[5|6]⟨2|(1+6)|5]s_234)
                   -1⟨6|(1+2)|3]³/(⟨1|2⟩⟨1|6⟩[3|4][4|5]⟨2|(1+6)|5]s_345)"""),
     ("ppmpmm", """+(-1⟨3|5⟩⁴[1|2]³)/(s_345[1|6]⟨4|5⟩⟨3|4⟩⟨3|(1+2)|6]⟨5|(1+6)|2])
                   +(+1[2|4]⁴⟨5|6⟩³)/(s_234⟨1|6⟩[2|3][3|4]⟨1|(2+3)|4]⟨5|(1+6)|2])
                   +(+1⟨3|(1+2)|4]⁴)/(⟨1|2⟩⟨2|3⟩[4|5][5|6]⟨1|(2+3)|4]⟨3|(1+2)|6]s_123)"""),
     ("pmpmpm", """+(+1⟨2|(1+3)|5]⁴)/(⟨1|2⟩⟨2|3⟩[4|5][5|6]⟨1|(2+3)|4]⟨3|(1+2)|6]s_123)
                   +(+1⟨6|(2+4)|3]⁴)/(⟨1|6⟩[2|3][3|4]⟨5|6⟩⟨1|(2+3)|4]⟨5|(1+6)|2]s_234)
                   +(-1⟨4|(3+5)|1]⁴)/([1|2][1|6]⟨3|4⟩⟨4|5⟩⟨3|(1+2)|6]⟨5|(1+6)|2]s_345)"""),
     ("pppppmm", """(⟨6|7⟩⁴)/(⟨1|2⟩⟨2|3⟩⟨3|4⟩⟨4|5⟩⟨5|6⟩⟨6|7⟩⟨7|1⟩)"""),
     ("ppppmmm", """+(-1⟨5|(6+7)|1]³)/([1|7]⟨2|3⟩⟨3|4⟩⟨4|5⟩[6|7]⟨2|(1+7)|6]s_167)
                    +(-1⟨7|(5+6)|4]³)/(⟨1|2⟩⟨1|7⟩⟨2|3⟩[4|5][5|6]⟨3|(4+5)|6]s_456)
                    +(-1⟨5|(3+4)|(1+2)|7⟩³)/(⟨1|2⟩⟨1|7⟩⟨3|4⟩⟨4|5⟩⟨2|(1+7)|6]⟨3|(4+5)|6]s_127s_345)"""),
     ("ppmpmpm", """+(-1[2|4]⁴⟨5|7⟩⁴)/(⟨1|7⟩[2|3][3|4]⟨5|6⟩⟨6|7⟩⟨1|(2+3)|4]⟨5|(3+4)|2]s_234)
                    +(-1⟨3|7⟩⁴[4|6]⁴)/(⟨1|2⟩⟨1|7⟩⟨2|3⟩[4|5][5|6]⟨3|(4+5)|6]⟨7|(5+6)|4]s_456)
                    +(-1[1|6]⁴⟨3|5⟩⁴)/([1|7]⟨2|3⟩⟨3|4⟩⟨4|5⟩[6|7]⟨2|(1+7)|6]⟨5|(6+7)|1]s_167)
                    +(+1⟨5|7⟩⁴⟨3|(1+2)|4]⁴)/(⟨1|2⟩⟨2|3⟩⟨5|6⟩⟨6|7⟩⟨1|(2+3)|4]⟨7|(5+6)|4]⟨3|(1+2)|(6+7)|5⟩s_123s_567)
                    +(-1⟨3|5⟩⁴⟨7|(1+2)|6]⁴)/(⟨1|2⟩⟨1|7⟩⟨3|4⟩⟨4|5⟩⟨2|(1+7)|6]⟨3|(4+5)|6]⟨5|(3+4)|(1+2)|7⟩s_127s_345)
                    +(-1[1|2]³⟨3|5⟩⁴⟨5|7⟩⁴)/(⟨3|4⟩⟨4|5⟩⟨5|6⟩⟨6|7⟩⟨5|(6+7)|1]⟨5|(3+4)|2]⟨3|(1+2)|(6+7)|5⟩⟨5|(3+4)|(1+2)|7⟩)""")]
)
def test_YM_with_analytics(helconf, analytic_expr):
    oParticles = lips.Particles(len(helconf))
    YMTreeAmp = seampy.NumericalAmplitude(theory="YM", helconf=helconf)
    assert(numpy.isclose(complex(YMTreeAmp(oParticles)), 1j * complex(oParticles(analytic_expr))) or
           numpy.isclose(complex(YMTreeAmp(oParticles)), -1j * complex(oParticles(analytic_expr))))


@pytest.mark.parametrize(
    ("helconf", ), [("ppmm", ), ("pmpm", )]
)
def test_YM_BS_EG_double_copy(helconf):
    oParticles = lips.Particles(len(helconf), seed=0)
    BSAmp = seampy.NumericalAmplitude(theory="BS", multiplicity=len(helconf))
    YMAmp = seampy.NumericalAmplitude(theory="YM", helconf=helconf)
    EGAmp = seampy.NumericalAmplitude(theory="EG", helconf=helconf)
    assert numpy.isclose(complex(YMAmp(oParticles) ** 2 / BSAmp(oParticles)), complex(EGAmp(oParticles)))
