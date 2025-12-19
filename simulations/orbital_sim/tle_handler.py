#!/usr/bin/env python3
"""
TLE Handler Module
Handles Two-Line Element set operations and conversions
"""

import numpy as np
from datetime import datetime
from typing import Dict, List, Optional


class TLEHandler:
    @staticmethod
    def keplerian_to_tle(name: str, orbital_elements: Dict, 
                         epoch: datetime = None, 
                         bstar: float = 0.0,
                         sat_num: int = 99999,
                         classification: str = 'U',
                         launch_year: int = None,
                         launch_num: int = 1,
                         piece: str = 'A') -> str:
        if epoch is None:
            epoch = datetime.utcnow()
        if launch_year is None:
            launch_year = epoch.year
        a = orbital_elements['a'] / 1000
        e = orbital_elements['e']
        i = orbital_elements['i']
        omega = orbital_elements['omega']
        w = orbital_elements['w']
        M = orbital_elements.get('M', 0)
        mu_earth = 398600.4418
        n = 86400 / (2 * np.pi * np.sqrt(a**3 / mu_earth))
        year = epoch.year % 100
        day_of_year = epoch.timetuple().tm_yday
        fraction_of_day = (epoch.hour * 3600 + epoch.minute * 60 + 
                          epoch.second + epoch.microsecond / 1e6) / 86400
        line0 = f"{name[:24]:24s}"
        line1 = "1 "
        line1 += f"{sat_num:05d}{classification} "
        line1 += f"{launch_year % 100:02d}{launch_num:03d}{piece:3s} "
        line1 += f"{year:02d}{day_of_year:03d}.{int(fraction_of_day * 100000000):08d} "
        line1 += " .00000000 "
        line1 += " 00000-0 "
        if bstar == 0:
            line1 += " 00000-0"
        else:
            exp = int(np.floor(np.log10(abs(bstar))))
            mantissa = bstar / (10 ** exp)
            sign = '+' if bstar >= 0 else '-'
            line1 += f" {sign}{abs(mantissa):.5f}".replace('.', '')[0:7]
            line1 += f"{exp:+d}"[-2:]
        line1 += " 0"
        line1 += "  999"
        line2 = "2 "
        line2 += f"{sat_num:05d} "
        line2 += f"{i:8.4f} "
        line2 += f"{omega:8.4f} "
        line2 += f"{int(e * 10000000):07d} "
        line2 += f"{w:8.4f} "
        line2 += f"{M:8.4f} "
        line2 += f"{n:11.8f}"
        line2 += "    1"
        line1 = TLEHandler._add_checksum(line1)
        line2 = TLEHandler._add_checksum(line2)
        return f"{line0}\\n{line1}\\n{line2}"
    
    @staticmethod
    def _add_checksum(line: str) -> str:
        checksum = 0
        for char in line:
            if char.isdigit():
                checksum += int(char)
            elif char == '-':
                checksum += 1
        return line + str(checksum % 10)
    
    @staticmethod
    def tle_to_keplerian(tle_lines: List[str]) -> Dict:
        if len(tle_lines) < 2:
            raise ValueError("Need at least 2 TLE lines")
        if len(tle_lines) == 3:
            line1, line2 = tle_lines[1], tle_lines[2]
            name = tle_lines[0].strip()
        else:
            line1, line2 = tle_lines[0], tle_lines[1]
            name = "UNKNOWN"
        if not line1.startswith('1 ') or not line2.startswith('2 '):
            raise ValueError("Invalid TLE format")
        sat_num = int(line1[2:7])
        classification = line1[7]
        launch_year = int(line1[9:11])
        launch_num = int(line1[11:14])
        piece = line1[14:17].strip()
        epoch_year = int(line1[18:20])
        epoch_day = float(line1[20:32])
        bstar_str = line1[53:61]
        if bstar_str.strip() == '00000-0':
            bstar = 0.0
        else:
            mantissa = float(bstar_str[0:6]) / 100000
            exp = int(bstar_str[6:8])
            bstar = mantissa * (10 ** exp)
        i = float(line2[8:16])
        omega = float(line2[17:25])
        e_str = line2[26:33]
        e = float('0.' + e_str)
        w = float(line2[34:42])
        M = float(line2[43:51])
        n = float(line2[52:63])
        rev_num = int(line2[63:68])
        mu_earth = 398600.4418
        n_rad_per_sec = n * 2 * np.pi / 86400
        a = (mu_earth / n_rad_per_sec**2)**(1/3) * 1000
        if epoch_year < 57:
            full_year = 2000 + epoch_year
        else:
            full_year = 1900 + epoch_year
        elements = {
            'name': name,
            'sat_num': sat_num,
            'a': a,
            'e': e,
            'i': i,
            'omega': omega,
            'w': w,
            'M': M,
            'n': n,
            'bstar': bstar,
            'epoch_year': full_year,
            'epoch_day': epoch_day,
            'rev_num': rev_num
        }
        return elements
    
    @staticmethod
    def validate_tle(tle_lines: List[str]) -> bool:
        if len(tle_lines) < 2:
            return False
        if len(tle_lines) == 3:
            line1, line2 = tle_lines[1], tle_lines[2]
        else:
            line1, line2 = tle_lines[0], tle_lines[1]
        if len(line1) != 69 or len(line2) != 69:
            return False
        if not line1.startswith('1 ') or not line2.startswith('2 '):
            return False
        for line in [line1, line2]:
            checksum = 0
            for char in line[:-1]:
                if char.isdigit():
                    checksum += int(char)
                elif char == '-':
                    checksum += 1
            if str(checksum % 10) != line[-1]:
                return False
        return True
    
    @staticmethod
    def read_tle_file(filename: str) -> List[Dict]:
        satellites = []
        with open(filename, 'r') as f:
            lines = f.readlines()
        i = 0
        while i < len(lines):
            if not lines[i].strip():
                i += 1
                continue
            if i + 2 < len(lines) and lines[i+1].startswith('1 ') and lines[i+2].startswith('2 '):
                tle_lines = [lines[i].strip(), lines[i+1].strip(), lines[i+2].strip()]
                i += 3
            elif i + 1 < len(lines) and lines[i].startswith('1 ') and lines[i+1].startswith('2 '):
                tle_lines = [lines[i].strip(), lines[i+1].strip()]
                i += 2
            else:
                i += 1
                continue
            if TLEHandler.validate_tle(tle_lines):
                try:
                    elements = TLEHandler.tle_to_keplerian(tle_lines)
                    satellites.append(elements)
                except Exception as e:
                    print(f"Error parsing TLE: {e}")
        return satellites
