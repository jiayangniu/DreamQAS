# Hamiltonian data

The six small NPZ inputs below are carried forward unchanged from the reference
release. `manifest.json` records exact SHA-256 hashes and sizes. Hash validation
establishes file identity; it is not a claim that molecular integrals have been
independently regenerated in this maintenance release.

| Filename | Bytes |
|---|---|
| `BEH2_6q_geom_H_0._0._-1.33;_Be_0._0._0.;_H_0._0._1.33_jordan_wigner.npz` | 68412 |
| `BEH2_8q_geom_Be_.0_.0_0.0;_H_.0_.0_1.326;_H_.0_.0_-1.326_jordan_wigner.npz` | 1058380 |
| `H2O_8q_geom_H_-0.021,_-0.002,_.0;_O_0.835,_0.452,_0;_H_1.477,_-0.273,_0_jordan_wigner.npz` | 1053200 |
| `H2O_8q_geom_O_.0_.0_0.0;_H_.0_0.757_0.586;_H_.0_-0.757_0.586_jordan_wigner.npz` | 1074794 |
| `LiH_4q_geom_Li_.0_.0_.0;_H_.0_.0_3.4_parity.npz` | 4808 |
| `LiH_6q_geom_Li_.0_.0_.0;_H_.0_.0_2.2_jordan_wigner.npz` | 36200 |

The default configurations cover LiH (4q/6q), BeH2 (6q/8q), and H2O (8q).
Two H2O geometries are retained; the selected configuration determines the filename.
NPZ files provide the Hamiltonian and reference/shift fields consumed by the
environment. Preserve the mapping, geometry, basis/active-space convention, and
energy shift when comparing against another benchmark. Filenames alone are not
a complete ab-initio generation recipe.

BeH2-10q and BeH2-12q configuration files are retained for compatibility, but their
Hamiltonians are not distributed in this Git repository. To run those tasks, obtain
the matching original inputs and point `DREAMQAS_MOL_DATA` to their directory.
A missing input causes a clear error with the required filename. There is no automatic
download or substituted molecule. Exact independent regeneration of the large-system
inputs requires additional basis/active-space provenance not supplied by this release.

Raw runs, checkpoints, optimizer traces, and complete comparison campaigns are
excluded. Generated data belongs in `runs/` or an explicitly chosen external disk.
