# Robot assets

`menagerie/` contains the KUKA LBR iiwa 14 and Unitree G1 robot descriptions,
collision/visual meshes, scene definitions, documentation and original licenses
from [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie).

The source commit and per-file SHA-256 hashes are recorded in
`menagerie/manifest.json`. Each model retains its own `LICENSE`; the project does
not replace those terms or claim manufacturer certification.

Setup is explicit: `python scripts/fetch_robot_assets.py` uses the recorded commit
on subsequent runs. Runtime checks the manifest and performs no downloads.
These files are additional project assets, not covered by the older migration
manifest for the four original explorations.

MJCF is used as the input model format for Newton's importer. Dynamics execution
uses Newton SolverMuJoCo with an explicitly selected CPU or CUDA numerical
backend, not PhysX. Tests against the original MJCF verify aspects of the import;
they do not establish physical accuracy against a real robot.
