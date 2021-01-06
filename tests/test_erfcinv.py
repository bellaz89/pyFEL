'''
    Test wheter the inverse error function matches the Genesis one
'''
from scipy.special import erfcinv
import numpy as np
from numpy import sqrt, log, exp

REL_ERROR = 1e-6
POINTS = 10000

def erfcinv_orig(y):
    z = y
    if y > 1:
        z = 2 - y

    w = 0.916461398268964 - log(z)
    u = sqrt(w)
    s = (log(u) + 0.488826640273108) / w
    t = 1 / (u + 0.231729200323405)
    x = (u * (1 - s * (s * 0.124610454613712 + 0.5)) - 
            + ((((-0.0728846765585675 * t + 0.269999308670029) * t + 
                + 0.150689047360223) * t + 0.116065025341614) * t + 
                + 0.499999303439796) * t)

    t = 3.97886080735226 / (x + 3.97886080735226)
    u = t - 0.5
    s = ((((((((((0.00112648096188977922 * u + 
        +   1.05739299623423047e-4) * u - 0.00351287146129100025) * u - 
        +   7.71708358954120939e-4) * u + 0.00685649426074558612) * u + 
        +   0.00339721910367775861) * u - 0.011274916933250487) * u - 
        +   0.0118598117047771104) * u + 0.0142961988697898018) * u + 
        +   0.0346494207789099922) * u + 0.00220995927012179067)
    s = (((((((((((((s * u - 0.0743424357241784861) * u - 
        +   0.105872177941595488) * u + 0.0147297938331485121) * u + 
        +   0.316847638520135944) * u + 0.713657635868730364) * u + 
        +   1.05375024970847138) * u + 1.21448730779995237) * u + 
        +   1.16374581931560831) * u + 0.956464974744799006) * u + 
        +   0.686265948274097816) * u + 0.434397492331430115) * u + 
        +   0.244044510593190935) * t - 
        +   z * exp(x * x - 0.120782237635245222))
    x = x+s * (x * s + 1)

    if y > 1:
        x = -x
    return x

def test_erfcinv():
    args = np.linspace(1.0/POINTS, 2.0-2.0/POINTS, POINTS)
    for arg in args:
        assert (erfcinv(arg) - erfcinv_orig(arg)) / erfcinv(arg) < REL_ERROR

if __name__ == "__main__":
    for i in range(10):
        print(erfcinv(i/10), erfcinv_orig(i/10)) 
