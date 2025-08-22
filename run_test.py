import inspect
import logging
import os
import sys
import numpy as np
from bioemu.sample import main as sample

logging.basicConfig(
    level=logging.INFO,
    stream=sys.stderr,     # 👈 send all log messages to stderr
    format="[%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

def clean_output_directory(out_dir):
    print(f"Cleaning {out_dir}")
    npz_files = sorted([os.path.join(out_dir, f) for f in os.listdir(out_dir) if f.endswith(".npz")])
    i = 1
    for f in npz_files:
        try:
            if os.path.exists(f):
                data = np.load(f)["pos"]
        except:
            print(f"BAD FILE: {f} - DELETING")
            os.remove(f)
            os.rename(npz_files[-i], f)
            i += 1


def mutate(in_sequence, mutations=None, offset=1):
    if mutations is None:
        mutations = []
    seq = list(in_sequence)
    for mut in mutations:
        in_res = mut[0]
        out_res = mut[-1]
        pos = int(mut[1:-1])
        idx = pos - offset
        if not (0 <= idx < len(seq)):
            raise IndexError(f"Position {pos} out of range for sequence length {len(seq)}")
        if seq[idx] == in_res:
            seq[idx] = out_res
        else:
            raise ValueError(f"{in_res}{pos} is not a residue in the given sequence")
    return "".join(seq)


# idx = sys.argv[1]
tem1_sequence = "HPETLVKVKDAEDQLGARVGYIELDLNSGKILESFRPEERFPMMSTFKVLLCGAVLSRVDAGQEQLGRRIHYSQNDLVEYSPVTEKHLTDGMTVRELCSAAITMSDNTAANLLLTTIGGPKELTAFLHNMGDHVTRLDRWEPELNEAIPNDERDTTMPAAMATTLRKLLTGELLTLASRQQLIDWMEADKVAGPLLRSALPAGWFIADKSGAGERGSRGIIAALGPDGKPSRIVVIYTTGSQATMDERNRQIAEIGASLIKHW"
mutations = ["A42G", "E104K", "M182T", "G238S"]
sequence = mutate(tem1_sequence, mutations, offset=26)
out_dir = "./blac_" + "_".join(mutations)
logger.info("OUTDIR: %s", out_dir)
sample(sequence=sequence, num_samples=2000, batch_size_100=20, output_dir=out_dir)
# clean_output_directory(out_dir)