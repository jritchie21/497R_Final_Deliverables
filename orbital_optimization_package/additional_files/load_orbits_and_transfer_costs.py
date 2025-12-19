import math
import os
from typing import Dict, Any, List
# from hohmann_transfer import hohmann_transfer
from max_hohmann import max_cost

####VARIABLES####
mu = 398600.0 #km^3/s^2


# ====== CLASSES ======
class Orbit:  # Simple orbit representation for circular orbits
    def __init__(self, orbit_number: int, description: str, r: float, i: float = 0.0, raan: float = 0.0, value: float = 0.0):
        """Initialize a circular orbit."""
        self.orbit_number = orbit_number
        self.description = description
        self.r = r
        self.i = i
        self.raan = raan
        self.value = value  # Orbit value/priority
    
    def __repr__(self):
        return f"Orbit({self.orbit_number}, {self.description}, r={self.r:.0f}, i={self.i:.1f}°, raan={self.raan:.1f}°, value={self.value:.1f})"


# ====== FILE LOADING FUNCTIONS ======
def load_orbits_from_file(filename: str, earth_radius: float = 6371.0, mu: float = 398600.0) -> List[Orbit]:
    """
    Load orbits from a configuration file.
    
    File format (each line):
        orbit_number, description, altitude_km, inclination_deg, raan_deg, value
    Or comments starting with #

    Returns:
        List of Orbit objects
    """
    orbits = []
    
    if not os.path.exists(filename):
        print(f"Warning: File {filename} not found!")
        return orbits
    
    with open(filename, 'r') as f:
        for line_num, line in enumerate(f, 1):
            # Strip whitespace
            line = line.strip()
            
            # Skip empty lines and comments
            if not line or line.startswith('#'):
                continue
            
            # Parse line:  orbit_number, description, altitude, inclination, value
            try:
                parts = [p.strip() for p in line.split(',')]
                if len(parts) < 5:
                    continue
                
                # Check if first column is a number (orbit_number)
                if parts[0].isdigit():
                    # Format: orbit_number, description, altitude, inclination, value
                    orbit_number = int(parts[0])
                    description = parts[1]
                    altitude = float(parts[2])
                    inclination = float(parts[3])
                    raan = float(parts[4])
                    value = float(parts[5]) if len(parts) >= 6 else 0.0
                    
                    # Calculate orbital radius (altitude + Earth radius)
                    radius = altitude + earth_radius
                    
                    # Create orbit
                    orbit = Orbit(orbit_number=orbit_number, description=description, r=radius, i=inclination, raan=raan, value=value)
                    orbits.append(orbit)
            except (ValueError, IndexError) as e:
                print(f"Warning: Could not parse line {line_num}: {line}")
                continue
    
    print(f"Loaded {len(orbits)} orbits from {filename}")
    return orbits


class TransferCosts:
    """
    Manages a network of orbits and computes transfer costs between them.
    """
    
    def __init__(self, mu: float = 398600.0):
        """Initialize transfer cost network."""
        self.orbits = []
        self.mu = mu
        self.cost_matrix = {}
        self.transfer_method = None
    
    def add_orbit(self, orbit: Orbit, quiet: bool = False) -> None:
        """Add an orbit to the network."""
        orbit.mu = self.mu  # Ensure consistent mu
        self.orbits.append(orbit)
    
    def load_from_file(self, filename: str, earth_radius: float = 6371.0) -> None:
        """Load orbits from a file into this network."""
        orbits = load_orbits_from_file(filename, earth_radius, self.mu)
        for orbit in orbits:
            self.add_orbit(orbit, quiet=True)
        print(f"Added {len(orbits)} orbits from {filename}")
    
    def compute_all_transfers(self, method: str = "max_hohmann") -> None:
        """Compute transfers between all orbit pairs using specified method."""
        self.transfer_method = method
        self.cost_matrix = {}
        n = len(self.orbits)

        from max_hohmann import max_cost
        
        # # Import transfer functions as needed
        # if method == "hohmann":
        #     from hohmann_transfer import hohmann_transfer, hohmann_with_plane_change
        #     coplanar_transfer = hohmann_transfer
        #     plane_change_transfer = hohmann_with_plane_change
        # elif method == "bielliptic":
        #     from bielliptic_transfer import bielliptic_transfer, bielliptic_with_plane_change
        #     coplanar_transfer = bielliptic_transfer
        #     plane_change_transfer = bielliptic_with_plane_change
        # elif method == "edelbaum":
        #     from edelbaum_transfer import edelbaum_transfer, edelbaum_with_plane_change
        #     coplanar_transfer = edelbaum_transfer
        #     plane_change_transfer = edelbaum_with_plane_change
        # elif method == "lambert":
        #     from lambert_transfer import lambert_transfer, lambert_with_plane_change
        #     coplanar_transfer = lambert_transfer
        #     plane_change_transfer = lambert_with_plane_change
        # else:
        #     print(f"Unknown transfer method: {method}. Using Hohmann.")
        #     from hohmann_transfer import hohmann_transfer, hohmann_with_plane_change
        #     coplanar_transfer = hohmann_transfer
        #     plane_change_transfer = hohmann_with_plane_change
        
        for i in range(n):
            for j in range(n):
                if i != j:  # Don't compute transfer to self
                    orbit1 = self.orbits[i]
                    orbit2 = self.orbits[j]
                    
                    result = max_cost(self.mu, orbit1.r, orbit1.i, orbit1.raan, orbit2.r, orbit2.i, orbit2.raan)
                    
                    # Add orbit info to result
                    result['from'] = orbit1.description
                    result['to'] = orbit2.description
                    
                    self.cost_matrix[(i, j)] = result
    
    def get_transfer(self, from_idx: int, to_idx: int):
        """Get transfer cost between two orbits by index."""
        return self.cost_matrix.get((from_idx, to_idx))
    
    def print_matrix(self) -> None:
        """Print a cost matrix table."""
        n = len(self.orbits)
        if n == 0:
            print("No orbits in network.")
            return
        
        # Header
        method_name = getattr(self, 'transfer_method', 'max_hohmann').title()
        print("\n" + "="*160)
        print(f"TRANSFER COST MATRIX ({method_name.capitalize()} Transfers)")
        print("="*160)
        print(f"\nNumber of orbits: {n}\n")
        
        # Create short labels for header
        short_labels = []
        for i, orbit in enumerate(self.orbits):
            short_labels.append(f"#{i+1}")
        
        # Print header row
        header = f"{'From\\To':<20}"
        for i, label in enumerate(short_labels):
            header += f"{label:>14}"
        print(header)
        
        # Print orbit info row
        info_row = f"{'':<20}"
        for orbit in self.orbits:
            info_row += f"{orbit.r:.0f}km".rjust(14)
        print(info_row)

                # Print orbit info row
        info_row2 = f"{'':<20}"
        for orbit in self.orbits:
            info_row2 += f"{orbit.i:.1f}i,{orbit.raan:.1f}rn".rjust(14)
        print(info_row2)
        
        # Print separator
        print("-"*160)
        
        # Print data rows - each cell shows both dv and time
        for i in range(n):
            row = f"{self.orbits[i].description:<20}"
            for j in range(n):
                if i == j:
                    row += f"{'---':>14}"
                else:
                    result = self.get_transfer(i, j)
                    if result:
                        # Format: "dv km/s, time h"
                        cell = f"{result['dv']:.2f}, {result['time']/3600:.1f}h"
                        row += cell.rjust(14)
            print(row)
        
        # Print legend below
        print("\n" + "-"*160)
        print("LEGEND:")
        for i, orbit in enumerate(self.orbits):
            print(f"  #{i+1}: {orbit.description} (r={orbit.r:.0f} km, i={orbit.i:.1f}°, raan={orbit.raan:.1f}°, value={orbit.value:.1f})")
        print("\nValues in matrix: Delta-V (km/s), Time (hours)")
        print("Radius is from center of earth, inclination and raan are in degrees")
        print("Earth radius is 6371 km")
        print("="*160)



# ====== EXAMPLE USAGE ======
if __name__ == "__main__":
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Display orbital transfer cost matrix",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python load_orbits_and_transfer_costs.py
  python load_orbits_and_transfer_costs.py orbits.txt edelbaum
  python load_orbits_and_transfer_costs.py orbits.txt --method hohmann
  python load_orbits_and_transfer_costs.py --method bielliptic
  python load_orbits_and_transfer_costs.py --file orbits.txt --method lambert
        """
    )
    parser.add_argument("orbits_file", nargs="?", type=str, default="orbits.txt",
                        help="Path to orbits configuration file (default: orbits.txt)")
    parser.add_argument("method_pos", nargs="?", type=str, choices=["hohmann", "bielliptic", "edelbaum", "lambert"],
                        default=None, help="Transfer method to use (positional argument)")
    parser.add_argument("--file", type=str, default=None,
                        help="Path to orbits configuration file (alternative to positional argument)")
    parser.add_argument("--method", type=str, choices=["hohmann", "bielliptic", "edelbaum", "lambert"],
                        default=None, help="Transfer method to use (alternative to positional argument)")
    
    args = parser.parse_args()
    
    # Determine orbits_file: --file flag takes precedence, then positional, then default
    orbits_file = args.file if args.file is not None else args.orbits_file
    
    # Determine method: --method flag takes precedence, then positional, then default to "hohmann"
    if args.method is not None:
        method = args.method
    elif args.method_pos is not None:
        method = args.method_pos
    else:
        method = "hohmann"
    
    print("=" * 80)
    print("ORBIT NETWORK TRANSFER ANALYSIS")
    print("=" * 80)
    
    # Create a network of orbits
    network = TransferCosts()
    
    # Option 1: Load from file
    if os.path.exists(orbits_file):
        print(f"\n--- Loading orbits from file: {orbits_file} ---")
        network.load_from_file(orbits_file)
            # Compute all transfers
        print(f"\nComputing all {method} transfers...")
        network.compute_all_transfers(method=method)
            # Print cost matrix
        network.print_matrix()
    else:
        print(f"\n--- File {orbits_file} not found")
