from collections import defaultdict
import os
import subprocess
import tempfile

from quartz.data import Grid


class WfnAnalysis:
    def __init__(self, wfn_file: str, mwfn_path: str):
        self.wfn_file = wfn_file  # fchk
        self.mwfn_path = mwfn_path
        self.temp_dir = tempfile.mkdtemp(prefix="temp_mwfn_")
        print(self.temp_dir)
        self.props = defaultdict(list)

        if os.environ.get("Multiwfnpath") or os.path.exists(
            os.path.join(os.getcwd(), "settings.ini")
        ):
            self.mwfn_init = True
        else:
            self.mwfn_init = False
        self.split_so_wfn = self.split_so_alpha_beta()

    def _process_mwfn_seq(self, input_seq):
        if self.mwfn_init:
            return "\n".join(input_seq)
        else:
            return "\n".join([""] + input_seq)

    def _get_mwfn(self) -> subprocess.Popen[str]:
        """Starts Multiwfn and provides pipe connection to it.

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
        Necessary for all wavefunctions from Psi4.
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
            mwfn.stdin.write(
                "\n"
            )  # FIXME - This needs to go as we expect settings.ini to be present
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

        # FIXME - Same here with ['']
        full_input_commands = self._process_mwfn_seq(
            [self.wfn_file] + input_seq + atom_inputs + ["0", "q"]
        )
        # print("Full input sequence for Multiwfn:\n", "\n".join(full_input_commands))
        # full_input_str = "\n".join(full_input_commands) + "\n"
        # print("Full input string for Multiwfn:\n", full_input_str)

        with self._get_mwfn() as mwfn:
            outs, errs = mwfn.communicate(input=full_input_commands)
        print("Multiwfn output: ", outs)
        print("Multiwfn errors: ", errs)

        lines = [l.strip() for l in outs.splitlines()]

        # TODO - probably would benefit from regex
        try:
            pol_header = (
                lines.index(
                    "Atomic polarizabilities estimated using Tkatchenko-Scheffler method:"
                )
                + 1
            )
            pol_end = lines.index("", pol_header) - 1

            for line in lines[pol_header:pol_end]:
                parts = line.split()
                self.props["polarizability"].append(float(parts[2]))

            c6_header = (
                lines.index(
                    "Atomic C6 coefficients estimated using Tkatchenko-Scheffler method:"
                )
                + 1
            )
            c6_end = lines.index("", c6_header)

            for line in lines[c6_header:c6_end]:
                parts = line.split()
                self.props["c6"].append(float(parts[2]))

        except Exception as e:
            raise Exception("Failed to parse polarizability/C6 results.") from e

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
            self.props["HOMO_contrib"].append(float(line.split()[4]) / 100)

        lumo_start = lines.index(" Composition of each atom:", homo_end) + 1
        lumo_end = lines.index("", lumo_start)

        for line in lines[lumo_start:lumo_end]:
            self.props["LUMO_contrib"].append(float(line.split()[4]) / 100)
