from shutil import rmtree
from collections import defaultdict
import os
import re
import subprocess
import tempfile

from quartz.data import Grid
from quartz.utils import angstrom_to_bohr


class WfnAnalysis:
    def __init__(self, wfn_file: str, mwfn_path: str):
        self.wfn_file = wfn_file  # fchk
        self.mwfn_path = mwfn_path
        self.temp_dir = tempfile.mkdtemp(prefix="temp_mwfn_")
        self.props = defaultdict(list)

        if os.environ.get("Multiwfnpath") or os.path.exists(
            os.path.join(os.getcwd(), "settings.ini")
        ):
            self.mwfn_init = True
        else:
            self.mwfn_init = False

        self.split_so_wfn = self.split_so_alpha_beta()
        self.decimal_regex = re.compile(
            r"\s*\d+\([A-Za-z]+\s*\)\s*:\s*([0-9]*\.?[0-9]+)"
        )
        # Format: Atom    1(C )  Effective V:    20.173  Free V:    34.583 a.u.  Ratio: 0.583
        self.volume_line_regex = re.compile(
            r"^Atom\s+\d+\([A-Za-z]+\s*\)\s+Effective V:\s+([0-9]*\.?[0-9]+)\s+Free V:\s+([0-9]*\.?[0-9]+)"
        )
        # Format: 1(C ):   6.591 a.u.  Contribution:  2.53 %  (Ref. data:  11.300 a.u.)
        self.pol_c6_line_regex = re.compile(
            r"\s*\d+\([A-Za-z]+\s*\):\s+([0-9]*\.?[0-9]+)\s+a\.u\."
        )
        # Format: 28(C )  -0.0696188369
        self.resp_regex = re.compile(r"\s*\d+\([a-zA-Z]+\s*\)\s+(\-*\d+\.\d+)")

        # print(f"Original wfn:\t{self.wfn_file}")
        # print(f"Temp dir:\t{self.temp_dir}")
        # print(f"Split wfn:\t{self.split_so_wfn}")

    def clean(self):
        rmtree(self.temp_dir)

    def _process_mwfn_seq(self, input_seq):
        if self.mwfn_init:
            return "\n".join(input_seq)
        else:
            return "\n".join([""] + input_seq)

    def _get_mwfn(self) -> subprocess.Popen[str]:
        """Starts Multiwfn and provides pipe connection to the process.

        :raises Exception: Multiwfn executable not found
        :return: Popen object for interfacing with Multiwfn
        :rtype: subprocess.Popen[str]
        """
        try:
            mwfn = subprocess.Popen(
                [self.mwfn_path],
                cwd=self.temp_dir,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            return mwfn
        except Exception:
            raise Exception(
                "Failed to launch Multiwfn. Ensure it is installed and in your PATH."
            )

    def split_so_alpha_beta(self):
        """
        Use Multiwfn to split spatial orbitals from fchk into alpha and beta spin orbitals.
        Necessary for all wavefunctions from Psi4. Hopefully it doesn't change the results.
        6 -> 37 -> 0 (Save to new.wfn & Return) -> q (Quit)
        """
        input_seq = self._process_mwfn_seq([self.wfn_file, "6", "37", "0", "q"])
        print(input_seq)
        with self._get_mwfn() as mwfn:
            outs, errs = mwfn.communicate(input=input_seq)
            print("Multiwfn output: ", outs)
            # print("Multiwfn errors: ", errs)
        # self.wfn_file = os.path.join(self.temp_dir, "new.wfn")
        return os.path.join(self.temp_dir, "new.wfn")

    def polarizability(self, atom_wfns_dir):
        """
        Use Multiwfn to calculate atomic polarizability from fchk file.
        Needs isolated atom wavefunctions (fchk format) and SO split (I think).
        15 -> 13 -> [atom wfn paths] -> 0 -> q
        """
        elements = []
        with self._get_mwfn() as mwfn:
            input_str = f"{self.split_so_wfn}\nq\n"
            outs, errs = mwfn.communicate(input=input_str)

            try:
                split_out = outs.split()
                if "Formula:" in split_out:
                    start_idx = split_out.index("Formula:") + 1
                    if "Total" in split_out[start_idx:]:
                        end_idx = split_out.index("Total", start_idx)
                        formula = split_out[start_idx:end_idx]
                        elements = [
                            "".join(filter(str.isalpha, elem)) for elem in formula
                        ]
                        # print(f"Parsed elements from Multiwfn output: {elements}")
            except Exception as e:
                print(f"Warning: Failed to parse elements from Multiwfn output: {e}")

        if not elements:
            raise Exception("Could not determine elements from wavefunction file.")

        input_seq = ["15", "13"]
        atom_inputs = []
        for atom in elements:
            # Handle standard naming or specific paths
            atom_wfn = os.path.join(atom_wfns_dir, f"{atom}.fchk")
            if not os.path.exists(atom_wfn):
                raise Exception(
                    f"No wavefunction found for element {atom} in {atom_wfns_dir}"
                )
            atom_inputs.append(atom_wfn)

        full_input_commands = self._process_mwfn_seq(
            [self.split_so_wfn] + input_seq + atom_inputs + ["0", "q"]
        )
        # print("Full input sequence for Multiwfn:\n", "\n".join(full_input_commands))
        # full_input_str = "\n".join(full_input_commands) + "\n"
        # print("Full input string for Multiwfn:\n", full_input_str)

        with self._get_mwfn() as mwfn:
            outs, errs = mwfn.communicate(input=full_input_commands)
        print("Multiwfn output: ", outs)
        print("Multiwfn errors: ", errs)

        lines = [l.strip() for l in outs.splitlines()]

        # Single-pass parsing: volumes, polarizabilities, and C6 coefficients
        section = None  # tracks which section we're in: 'pol' or 'c6'
        curr_line = None
        try:
            for line in lines:
                curr_line = line

                # Volume lines can appear before the section headers
                vol_match = self.volume_line_regex.search(line)
                if vol_match:
                    self.props["effectiveV"].append(float(vol_match.group(1)))
                    self.props["freeV"].append(float(vol_match.group(2)))
                    continue

                # Detect section headers
                if (
                    "Atomic polarizabilities estimated using Tkatchenko-Scheffler method:"
                    in line
                ):
                    section = "pol"
                    continue
                if (
                    "Atomic C6 coefficients estimated using Tkatchenko-Scheffler method:"
                    in line
                ):
                    section = "c6"
                    continue

                # Parse polarizability / C6 data lines
                if section in ("pol", "c6"):
                    match = self.pol_c6_line_regex.match(line)
                    if match:
                        key = "polarizability" if section == "pol" else "c6"
                        self.props[key].append(float(match.group(1)))
                    elif not line:  # blank line ends the section
                        section = None

        except Exception as e:
            print(
                f"Error parsing volume/polarizability/C6 results at line: '{curr_line}'"
            )
            raise Exception("Failed to parse volume/polarizability/C6 results.") from e

        if len(self.props["effectiveV"]) != len(self.props["freeV"]):
            raise Exception(
                f"Mismatch in number of effective volumes ({len(self.props['effectiveV'])}) and free volumes ({len(self.props['freeV'])}) parsed."
            )

        if len(self.props["polarizability"]) != len(self.props["c6"]):
            raise Exception(
                f"Mismatch in number of polarizabilities ({len(self.props['polarizability'])}) and C6 coefficients ({len(self.props['c6'])}) calculated."
            )

    def grid_properties(self, reference_grid: Grid):
        """
        Calculate properties on a grid with Multiwfn. Use reference Grid object for grid origin, spacing and cell count.
        """
        pass

    def homo_lumo_atom_contribs(self, partition: str = "Mul"):
        """
        HOMO and LUMO densities/localizations on atoms.
        8 -> (1, 2, 3 depending on partitioning scheme) -> [h, l] -> 0 -> -10 -> q
        Contribs returned in %, divide by 100 for values from 0 to 1
        """
        partition_keys = {
            "Mul": 1,  # Mulliken
            "SP": 2,  # Stout-Politzer
            "SCPA": 3,  # Ros-Schuit
        }
        input_seq = self._process_mwfn_seq(
            [
                self.wfn_file,
                "8",
                str(partition_keys[partition]),
                "h",
                "l",
                "0",
                "-10",
                "q",
            ]
        )
        print(input_seq)
        with self._get_mwfn() as mwfn:
            outs, errs = mwfn.communicate(input=input_seq)
            print(outs)

        lines = outs.splitlines()

        homo_start = lines.index(" Composition of each atom:") + 1
        print(homo_start)
        homo_end = lines.index("", homo_start)
        print(homo_end)

        for line in lines[homo_start:homo_end]:
            match = self.decimal_regex.search(line)
            if match:
                self.props["HOMO_contrib"].append(float(match.group(1)) / 100)

        lumo_start = lines.index(" Composition of each atom:", homo_end) + 1
        lumo_end = lines.index("", lumo_start)

        for line in lines[lumo_start:lumo_end]:
            match = self.decimal_regex.search(line)
            if match:
                self.props["LUMO_contrib"].append(float(match.group(1)) / 100)

    def resp_charges(self):
        """
        7 -> 18 -> 1 -> 18
        check default values of a agains psiresp
        FIXME - breaks on Bromine and Iodine
        > vdW radius used in fitting for Br is missing, input it in Angstrom, e.g. 1.7
        > Hint: If pressing ENTER button directly, corresponding UFF radii scaled by 1/1.2 will be used, which usually is a proper workaround. (If you want to automatically employ this treatment, setting "ispecial" in settings.ini to 1)
        """
        input_seq = [
            self.wfn_file,
            "7",
            "18",
            "1",
        ]

        with self._get_mwfn() as mwfn:
            outs, errs = mwfn.communicate(input=self._process_mwfn_seq(input_seq))
            print(outs)

        lines = [line.strip() for line in outs.splitlines()]
        resp_start = lines.index("Center       Charge") + 1
        resp_end = lines.index("", resp_start) - 2

        for line in lines[resp_start:resp_end]:
            match = self.resp_regex.search(line)
            if match:
                self.props["RESP"].append(float(match.group(1)))
