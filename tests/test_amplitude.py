import lips
import seampy


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #


def test_amplitude():
    oParticles = lips.Particles(6)
    oParticles.fix_mom_cons()
    oNumAmp = seampy.NumericalAmplitude(theory="DF2", helconf="pppmmm")
    oNumAmp(oParticles)
