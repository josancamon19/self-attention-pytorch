#!/usr/bin/env python3
"""
Interactive Triton Debug Output Parser

Parses tl.device_print output and provides an interactive shell to explore the data.
Usage: python your_program.py 2>&1 | python debug_triton.py
   or: python debug_triton.py < output.log
"""

import sys
import re
from collections import defaultdict, namedtuple
import json
from typing import Dict, List, Tuple, Any
import argparse
import numpy as np
import code
import readline

# Data structures
ProgramInstance = namedtuple('ProgramInstance', ['pid0', 'pid1', 'pid2'])

class TritonData:
    """Main data container for parsed Triton debug output."""
    
    def __init__(self, tensor_data, scalar_data, program_info):
        self.tensor_data = tensor_data
        self.scalar_data = scalar_data
        self.program_info = program_info
        self._arrays = {}
        self._shapes = {}
        
        # Convert tensors to numpy arrays for easier manipulation
        for pid in tensor_data:
            self._arrays[pid] = {}
            self._shapes[pid] = {}
            for var_name in tensor_data[pid]:
                tensor_array = self._dict_to_array(tensor_data[pid][var_name])
                self._arrays[pid][var_name] = tensor_array
                self._shapes[pid][var_name] = tensor_array.shape
    
    def _dict_to_array(self, index_dict):
        """Convert {(i,j): value} dict to numpy array."""
        if not index_dict:
            return np.array([])
        
        indices = list(index_dict.keys())
        max_i = max(idx[0] for idx in indices) + 1
        max_j = max(idx[1] for idx in indices) + 1
        
        array = np.zeros((max_i, max_j))
        for (i, j), value in index_dict.items():
            array[i, j] = value
        
        return array
    
    def __getitem__(self, key):
        """Support data[pid] access."""
        if isinstance(key, (tuple, ProgramInstance)):
            if isinstance(key, tuple) and len(key) == 3:
                key = ProgramInstance(*key)
            tensor_arrays = self._arrays.get(key, {})
            tensor_shapes = self._shapes.get(key, {})
            scalar_values = self.scalar_data.get(key, {})
            return ProgramDict(tensor_arrays, tensor_shapes, scalar_values)
        return self._arrays.get(key, {})
    
    def keys(self):
        return self.program_info.keys()
    
    def program_ids(self):
        """Get all program IDs."""
        return list(self.program_info.keys())
    
    def variables(self):
        """Get all variable names across all programs."""
        vars_set = set()
        for pid_info in self.program_info.values():
            vars_set.update(pid_info['tensor_vars'])
            vars_set.update(pid_info['scalar_vars'])
        return list(vars_set)
    
    def summary(self):
        """Print comprehensive summary."""
        print("=== TRITON DEBUG SUMMARY ===")
        print(f"📊 Total program instances: {len(self.program_ids())}")
        print(f"📝 All variables: {', '.join(sorted(self.variables()))}")
        
        # Program ID breakdown by k, bh values
        k_values = set()
        bh_values = set()
        for pid in self.program_ids():
            k_values.add(pid.pid0)  # k = program_id(0)
            bh_values.add(pid.pid1)  # bh = program_id(1)
        
        print(f"\n🎯 Program ID breakdown:")
        print(f"   k values (pid0): {sorted(k_values)} ({len(k_values)} unique)")
        print(f"   bh values (pid1): {sorted(bh_values)} ({len(bh_values)} unique)")
        
        # Variable breakdown by type
        tensor_vars = set()
        scalar_vars = set()
        for pid_info in self.program_info.values():
            tensor_vars.update(pid_info['tensor_vars'])
            scalar_vars.update(pid_info['scalar_vars'])
        
        print(f"\n📊 Variable types:")
        if tensor_vars:
            print(f"   Tensors: {', '.join(sorted(tensor_vars))}")
        if scalar_vars:
            print(f"   Scalars: {', '.join(sorted(scalar_vars))}")
        
        # Show details for each program instance
        print(f"\n📋 Program instances:")
        for pid in sorted(self.program_ids()):
            info = self.program_info[pid]
            print(f"   {pid}: tensors={list(info['tensor_vars'])}, scalars={list(info['scalar_vars'])}")
    
    def shapes(self):
        """Print shapes of all tensors."""
        print("=== TENSOR SHAPES ===")
        for pid in sorted(self._arrays.keys()):
            print(f"\nProgram ID {pid}:")
            for var_name in sorted(self._arrays[pid].keys()):
                shape = self._shapes[pid][var_name]
                print(f"  {var_name}: {shape}")
    
    def scalars(self):
        """Print scalar values."""
        print("=== SCALAR VALUES ===")
        for pid in sorted(self.scalar_data.keys()):
            print(f"\nProgram ID {pid}:")
            for var_name in sorted(self.scalar_data[pid].keys()):
                values = self.scalar_data[pid][var_name]
                unique_values = list(set(values))
                if len(unique_values) == 1:
                    print(f"  {var_name}: {unique_values[0]} (repeated {len(values)} times)")
                else:
                    print(f"  {var_name}: {unique_values} (total {len(values)} values)")
    
    def get_by_k_bh(self, k=None, bh=None):
        """Get programs filtered by k and/or bh values."""
        matching_pids = []
        for pid in self.program_ids():
            if k is not None and pid.pid0 != k:
                continue
            if bh is not None and pid.pid1 != bh:
                continue
            matching_pids.append(pid)
        return matching_pids

class ProgramDict:
    """Dictionary for a specific program ID with tensor and scalar access."""
    
    def __init__(self, arrays, shapes, scalars):
        self._arrays = arrays
        self._shapes = shapes
        self._scalars = scalars
        
        # Add tensor variables as attributes for easy access
        for var_name, array in arrays.items():
            setattr(self, var_name, array)
        
        # Add scalar variables as attributes
        for var_name, values in scalars.items():
            # For scalars, use the first value or create summary
            if len(values) == 1:
                setattr(self, var_name, values[0])
            else:
                setattr(self, var_name, values)  # Keep as list if multiple values
    
    def __getitem__(self, key):
        if key in self._arrays:
            return self._arrays[key]
        elif key in self._scalars:
            return self._scalars[key]
        return None
    
    def keys(self):
        return list(self._arrays.keys()) + list(self._scalars.keys())
    
    def tensor_keys(self):
        return list(self._arrays.keys())
    
    def scalar_keys(self):
        return list(self._scalars.keys())
    
    def shapes(self):
        """Print shapes and scalars for this program."""
        if self._arrays:
            print("Tensors:")
            for var_name in sorted(self._arrays.keys()):
                shape = self._shapes[var_name]
                print(f"  {var_name}: {shape}")
        
        if self._scalars:
            print("Scalars:")
            for var_name in sorted(self._scalars.keys()):
                values = self._scalars[var_name]
                if len(set(values)) == 1:
                    print(f"  {var_name}: {values[0]}")
                else:
                    print(f"  {var_name}: {values} (multiple values)")

class TritonDebugParser:
    def __init__(self):
        # Pattern for tensors: pid (143, 0, 0) idx (58, 23) k_tile: 0.495480
        self.tensor_pattern = re.compile(
            r'pid\s*\((\d+),\s*(\d+),\s*(\d+)\)\s*idx\s*\((\d+),\s*(\d+)\)\s*(\w+):\s*([+-]?\d*\.?\d+(?:[eE][+-]?\d+)?)'
        )
        
        # Pattern for scalars: pid (0, 0, 0) idx () k_start: 16
        self.scalar_pattern = re.compile(
            r'pid\s*\((\d+),\s*(\d+),\s*(\d+)\)\s*idx\s*\(\)\s*(\w+):\s*([+-]?\d*\.?\d+(?:[eE][+-]?\d+)?)'
        )
        
        # Storage
        self.tensor_data = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
        self.scalar_data = defaultdict(lambda: defaultdict(list))  # scalars can have multiple values
        self.variables = set()
        self.program_instances = set()
        self.program_info = defaultdict(lambda: {'tensor_vars': set(), 'scalar_vars': set()})
        
    def parse_line(self, line: str) -> bool:
        """Parse a single line and extract debug info."""
        line = line.strip()
        
        # Try tensor pattern first
        tensor_match = self.tensor_pattern.match(line)
        if tensor_match:
            pid0, pid1, pid2, idx0, idx1, var_name, value = tensor_match.groups()
            program_id = ProgramInstance(int(pid0), int(pid1), int(pid2))
            index = (int(idx0), int(idx1))
            value = float(value)
            
            # Store tensor data
            self.tensor_data[program_id][var_name][index] = value
            self.variables.add(var_name)
            self.program_instances.add(program_id)
            self.program_info[program_id]['tensor_vars'].add(var_name)
            return True
        
        # Try scalar pattern
        scalar_match = self.scalar_pattern.match(line)
        if scalar_match:
            pid0, pid1, pid2, var_name, value = scalar_match.groups()
            program_id = ProgramInstance(int(pid0), int(pid1), int(pid2))
            value = float(value)
            
            # Store scalar data
            self.scalar_data[program_id][var_name].append(value)
            self.variables.add(var_name)
            self.program_instances.add(program_id)
            self.program_info[program_id]['scalar_vars'].add(var_name)
            return True
        
        return False
    
    def parse_stream(self, stream):
        """Parse entire input stream."""
        line_count = 0
        parsed_count = 0
        
        for line in stream:
            line_count += 1
            if self.parse_line(line):
                parsed_count += 1
                if parsed_count % 1000 == 0:
                    print(f"Parsed {parsed_count} debug lines...", file=sys.stderr)
        
        print(f"Total lines: {line_count}, Parsed: {parsed_count}", file=sys.stderr)
        return TritonData(self.tensor_data, self.scalar_data, self.program_info)

def start_interactive_shell(data):
    """Start interactive Python shell with triton data loaded."""
    
    # Create convenient aliases
    program_ids = data.program_ids()
    variables = data.variables()
    
    # Shortcuts for first program (most common case)
    if program_ids:
        first_pid = program_ids[0]
        prog = data[first_pid]
        
        # Add individual variables to namespace for easy access
        namespace = {
            'data': data,
            'summary': data.summary,
            'shapes': data.shapes,
            'scalars': data.scalars,
            'program_ids': program_ids,
            'variables': variables,
            'prog': prog,
            'get_by_k_bh': data.get_by_k_bh,
            'np': np,
        }
        
        # Add all variables from first program to global namespace
        for var_name in prog.keys():
            namespace[var_name] = getattr(prog, var_name)
    else:
        namespace = {
            'data': data,
            'summary': data.summary,
            'shapes': data.shapes,
            'scalars': data.scalars,
            'program_ids': program_ids,
            'variables': variables,
            'get_by_k_bh': data.get_by_k_bh,
            'np': np,
        }
    
    # Print welcome message
    print("\n" + "="*60)
    print("🔬 TRITON DEBUG INTERACTIVE SHELL")
    print("="*60)
    data.summary()
    
    if program_ids:
        print(f"\n🎯 Quick access (first program {program_ids[0]}):")
        prog.shapes()
    
    print(f"\n💡 Try these commands:")
    print(f"   summary()                   # Show comprehensive overview")
    print(f"   shapes()                    # Show all tensor shapes")
    print(f"   scalars()                   # Show all scalar values")
    print(f"   data[(0,0,0)].shapes()      # Shapes for specific program")
    print(f"   get_by_k_bh(k=0)            # Get programs where k=0")
    print(f"   get_by_k_bh(k=0, bh=1)      # Get programs where k=0 and bh=1")
    if variables:
        # Find a tensor variable for example
        tensor_vars = [v for v in variables if hasattr(prog, v) and hasattr(getattr(prog, v), 'shape')]
        if tensor_vars:
            var_example = tensor_vars[0]
            print(f"   {var_example}[:5, :5]           # Slice tensor data")
            print(f"   {var_example}.mean()            # Compute statistics")
    print(f"   program_ids                 # List all program IDs")
    print(f"   variables                   # List all variable names")
    print("="*60)
    
    # Start interactive shell
    code.InteractiveConsole(namespace).interact()

def main():
    parser = argparse.ArgumentParser(description='Interactive Triton debug parser')
    parser.add_argument('--summary-only', action='store_true', help='Just print summary and exit')
    parser.add_argument('--shapes-only', action='store_true', help='Just print shapes and exit')
    parser.add_argument('--scalars-only', action='store_true', help='Just print scalars and exit')
    parser.add_argument('--file', type=str, help='Read from file instead of stdin')
    args = parser.parse_args()
    
    # Parse input
    debug_parser = TritonDebugParser()
    
    if args.file:
        with open(args.file, 'r') as f:
            data = debug_parser.parse_stream(f)
    else:
        # Check if stdin has data
        if sys.stdin.isatty():
            print("Error: No input provided. Use --file or pipe input.", file=sys.stderr)
            print("Example: python your_program.py 2>&1 | python debug_triton.py", file=sys.stderr)
            sys.exit(1)
        data = debug_parser.parse_stream(sys.stdin)
    
    if args.summary_only:
        data.summary()
        return
    
    if args.shapes_only:
        data.shapes()
        return
        
    if args.scalars_only:
        data.scalars()
        return
    
    # For interactive shell, we need to restore stdin
    if not args.file:
        # Reopen stdin for interactive use
        sys.stdin = open('/dev/tty', 'r')
    
    # Start interactive shell
    start_interactive_shell(data)

if __name__ == "__main__":
    main() 