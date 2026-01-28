from math import pi

import trasyn
from trasyn.gates import t

def main():
    seq, mat, err = trasyn.synthesize(t(), nonclifford_budget=10)
    print(seq, err)
    seq, mat, err = trasyn.synthesize([0.1, 0.2, 0.3], nonclifford_budget=20) # U(0.1, 0.2, 0.3)
    print(seq, err, seq.count("t"))
    seq, mat, err = trasyn.synthesize(pi / 16, 30, error_threshold=0.001) # Rz(pi/16)
    print(seq, err, seq.count("t"))
    print(123)

if __name__ == "__main__":
    main()