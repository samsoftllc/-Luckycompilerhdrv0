# program.py
import tkinter as tk
from tkinter import ttk, scrolledtext, filedialog, messagebox
import os
import sys
import re
import struct

# --- Instruction Set Definition ---

# Opcodes (8-bit)
# Format (RR):  Opcode (8) | SrcReg (4) | DestReg (4)
# Format (RI):  Opcode (8) | DestReg (4) | 0000 | Immediate (16)
# Format (JMP): Opcode (8) | 0000 0000   | Target Address (16)
OPCODES = {
    # Register-Register
    'add': 0x00,
    'sub': 0x01,
    'mul': 0x02,
    'div': 0x03,
    'cmp': 0x04,
    'mov': 0x05,

    # Register-Immediate
    'mov_imm': 0x10,
    'add_imm': 0x11,
    'sub_imm': 0x12,
    'cmp_imm': 0x13,
    'mul_imm': 0x14,  # added
    'div_imm': 0x15,  # added

    # Jumps
    'jmp': 0x20,
    'je':  0x21,
    'jne': 0x22,
}

# Registers (4-bit codes)
REGISTERS = {
    'rax': 0x0A,
    'rbx': 0x0B,
    'rcx': 0x0C,
    'rdx': 0x0D,
}

class Assembler:
    def __init__(self):
        self.symbol_table = {}            # label -> address (byte offset)
        self.intermediate_code = []       # list of dicts with parsed instructions
        self.binary_code = bytearray()
        self.current_address = 0

    def _parse_operand(self, operand_str):
        """Detect operand type (register, immediate, label)."""
        operand_str = operand_str.strip()
        if not operand_str:
            return None, None

        low = operand_str.lower()

        # Register
        if low in REGISTERS:
            return 'register', low

        # Immediate (dec/hex, with negative support)
        try:
            if low.startswith('0x'):
                value = int(low, 16)
            else:
                value = int(low, 10)
            if 0 <= value <= 0xFFFF:
                return 'immediate', value
            elif -0x8000 <= value < 0:
                return 'immediate', value & 0xFFFF
            else:
                return 'error', f"Immediate value out of 16-bit range: {operand_str}"
        except ValueError:
            pass

        # Label (basic validation)
        if re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', operand_str):
            return 'label', operand_str
        return 'error', f"Invalid operand format: {operand_str}"

    def pass1(self, source_lines):
        """Parse, collect labels (by instruction index), and build IR."""
        self.messages = ["--- Pass 1: Parsing and Symbol Table ---"]
        instruction_index = 0

        for line_num, raw in enumerate(source_lines, 1):
            line = raw.strip()
            # strip comments
            cpos = line.find(';')
            if cpos != -1:
                line = line[:cpos].strip()
            if not line:
                continue

            # Label
            label_match = re.match(r'^([a-zA-Z_][a-zA-Z0-9_]*):\s*(.*)$', line)
            instruction_part = line
            if label_match:
                label, instruction_part = label_match.groups()
                label = label.lower()
                if label in self.symbol_table:
                    self.messages.append(f"Error line {line_num}: Label '{label}' redefined.")
                else:
                    self.symbol_table[label] = instruction_index
                    self.messages.append(f"  Found label '{label}' at index {instruction_index}")
                instruction_part = instruction_part.strip()

            if not instruction_part:
                continue

            # Instruction + operands
            parts = re.split(r'[,\s]+', instruction_part, maxsplit=2)
            opcode_str = parts[0].lower()
            operands = [p.strip() for p in parts[1:] if p.strip()]

            if not opcode_str:
                self.messages.append(f"Warning line {line_num}: Empty instruction part '{instruction_part}'")
                continue

            self.intermediate_code.append({
                'line': line_num,
                'opcode': opcode_str,
                'operands': operands,
                'index': instruction_index
            })
            instruction_index += 1

        self.messages.append(f"Symbol Table (indices): {self.symbol_table}")
        self.messages.append("--- Pass 1 Complete ---")
        return True

    def optimize(self):
        """Simple peephole optimizer."""
        self.messages.append("--- Optimization Pass ---")
        optimized = []
        removed = 0
        i = 0
        while i < len(self.intermediate_code):
            instr = self.intermediate_code[i]
            op = instr['opcode']
            ops = instr['operands']

            drop = False

            # Remove 'mov r, r'
            if op == 'mov' and len(ops) == 2:
                k1, v1 = self._parse_operand(ops[0])
                k2, v2 = self._parse_operand(ops[1])
                if k1 == k2 == 'register' and v1 == v2:
                    self.messages.append(f"  Optimizing line {instr['line']}: drop redundant 'mov {v1}, {v2}'")
                    drop = True
                    removed += 1

            # Remove consecutive duplicate 'mov dest, src'
            if not drop and op == 'mov' and i + 1 < len(self.intermediate_code):
                nxt = self.intermediate_code[i + 1]
                if nxt['opcode'] == 'mov' and nxt['operands'] == ops:
                    self.messages.append(f"  Optimizing lines {instr['line']} & {nxt['line']}: drop duplicate second mov")
                    # keep the first, drop the second
                    optimized.append(instr)
                    i += 2
                    removed += 1
                    continue

            if not drop:
                optimized.append(instr)
            i += 1

        self.intermediate_code = optimized

        # Remap label indices after removals
        index_map = {old['index']: i for i, old in enumerate(self.intermediate_code)}
        new_symbols = {}
        for label, old_idx in self.symbol_table.items():
            if old_idx in index_map:
                new_symbols[label] = index_map[old_idx]
            else:
                # find next surviving instruction
                next_new = -1
                for j in range(old_idx + 1, len(self.intermediate_code) + removed):
                    if j in index_map:
                        next_new = index_map[j]
                        break
                if next_new != -1:
                    new_symbols[label] = next_new
                    self.messages.append(f"  Warning: Label '{label}' remapped to index {next_new} (target optimized away)")
                else:
                    new_symbols[label] = len(self.intermediate_code)
                    self.messages.append(f"  Warning: Label '{label}' remapped past end (target optimized away at end)")

        self.symbol_table = new_symbols
        for i, instr in enumerate(self.intermediate_code):
            instr['index'] = i

        self.messages.append(f"Optimization removed {removed} instructions.")
        self.messages.append(f"Updated Symbol Table (indices): {self.symbol_table}")
        self.messages.append("--- Optimization Complete ---")

    def _get_instruction_size(self, opcode, operands):
        """Return encoded size in bytes, or -1 if unsupported."""
        op = opcode.lower()

        # RR: reg, reg (dest, src in source)
        if op in ('add', 'sub', 'mul', 'div', 'cmp', 'mov'):
            if len(operands) == 2:
                t1, _ = self._parse_operand(operands[0])
                t2, _ = self._parse_operand(operands[1])
                if t1 == 'register' and t2 == 'register':
                    return 2

        # RI: reg, imm (dest, imm)
        if op in ('mov', 'add', 'sub', 'cmp', 'mul', 'div'):
            if len(operands) == 2:
                t1, _ = self._parse_operand(operands[0])
                t2, _ = self._parse_operand(operands[1])
                if t1 == 'register' and t2 == 'immediate':
                    return 4

        # Jumps: label
        if op in ('jmp', 'je', 'jne'):
            if len(operands) == 1:
                t1, _ = self._parse_operand(operands[0])
                if t1 == 'label':
                    return 4

        return -1

    def pass2(self):
        """Compute byte addresses, resolve labels, and encode."""
        self.messages.append("--- Pass 2: Encoding and Address Resolution ---")
        self.binary_code = bytearray()
        self.current_address = 0
        errors = False

        # Instruction index -> byte address
        addr = 0
        instr_addr = {}
        for instr in self.intermediate_code:
            instr_addr[instr['index']] = addr
            size = self._get_instruction_size(instr['opcode'], instr['operands'])
            if size == -1:
                self.messages.append(f"Error line {instr['line']}: Unsupported form '{instr['opcode']} {' '.join(instr['operands'])}'")
                errors = True
                size = 0
            addr += size

        # Label -> byte address
        final_symbols = {}
        for label, idx in self.symbol_table.items():
            if idx in instr_addr:
                final_symbols[label] = instr_addr[idx]
            else:
                self.messages.append(f"Error: Label '{label}' index {idx} not found post-optimization.")
                final_symbols[label] = 0xFFFF
                errors = True
        self.symbol_table = final_symbols
        self.messages.append(f"Final Symbol Table (byte addresses): {self.symbol_table}")

        # Encode
        for instr in self.intermediate_code:
            op = instr['opcode']
            ops = instr['operands']
            line = instr['line']
            encoded = None

            try:
                # RR: dest, src  ->  fields: src | dest
                if op in ('add', 'sub', 'mul', 'div', 'cmp', 'mov') and len(ops) == 2:
                    t1, v1 = self._parse_operand(ops[0])
                    t2, v2 = self._parse_operand(ops[1])

                    if t1 == 'register' and t2 == 'register':
                        if op not in OPCODES:
                            raise ValueError(f"Internal error: opcode '{op}' not found")
                        op_byte = OPCODES[op]
                        src_reg = REGISTERS[v2]   # src is operand 2
                        dest_reg = REGISTERS[v1]  # dest is operand 1
                        word = (op_byte << 8) | (src_reg << 4) | dest_reg
                        encoded = struct.pack('>H', word)

                    # RI: dest, imm
                    elif t1 == 'register' and t2 == 'immediate':
                        imm_op = f"{op}_imm"
                        if imm_op not in OPCODES:
                            raise ValueError(f"Unsupported immediate operation: {op}")
                        op_byte = OPCODES[imm_op]
                        dest_reg = REGISTERS[v1]
                        part1 = (op_byte << 8) | (dest_reg << 4)
                        encoded = struct.pack('>H', part1) + struct.pack('>H', v2)

                    else:
                        raise ValueError(f"Invalid operand types for {op}: {t1}, {t2}")

                # JMP label
                elif op in ('jmp', 'je', 'jne') and len(ops) == 1:
                    t1, v1 = self._parse_operand(ops[0])
                    if t1 != 'label':
                        raise ValueError(f"Invalid operand for {op}: expected label, got {t1}")
                    if v1 not in self.symbol_table:
                        raise ValueError(f"Undefined label: '{v1}'")
                    target = self.symbol_table[v1]
                    if not (0 <= target <= 0xFFFF):
                        raise ValueError(f"Target address {target} out of range for label '{v1}'")
                    op_byte = OPCODES[op]
                    part1 = op_byte << 8
                    encoded = struct.pack('>H', part1) + struct.pack('>H', target)

                else:
                    raise ValueError(f"Unknown/invalid instruction: {op} {' '.join(ops)}")

            except ValueError as e:
                self.messages.append(f"Error line {line}: {e}")
                errors = True

            if encoded:
                self.binary_code.extend(encoded)
                self.current_address += len(encoded)

        self.messages.append(f"--- Pass 2 Complete ({len(self.binary_code)} bytes generated) ---")
        return not errors

    def write_output(self, filename):
        try:
            with open(filename, 'wb') as f:
                f.write(self.binary_code)
            self.messages.append(f"Successfully wrote {len(self.binary_code)} bytes to '{filename}'")
            return True
        except IOError as e:
            self.messages.append(f"Error writing output file '{filename}': {e}")
            return False

    def assemble_from_text(self, source_text, output_filename=None):
        """Assemble from text content instead of file."""
        self.symbol_table = {}
        self.intermediate_code = []
        self.binary_code = bytearray()
        self.current_address = 0
        self.messages = []
        
        source_lines = source_text.split('\n')
        
        if self.pass1(source_lines):
            self.optimize()
            if self.pass2():
                if output_filename:
                    success = self.write_output(output_filename)
                    return success, "\n".join(self.messages)
                return True, "\n".join(self.messages)
            else:
                return False, "\n".join(self.messages)
        else:
            return False, "\n".join(self.messages)


class AssemblerGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Assembler GUI")
        self.root.geometry("600x400")
        
        self.assembler = Assembler()
        self.current_file = None
        
        self.setup_ui()
        
    def setup_ui(self):
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(1, weight=1)
        
        # File operations frame
        file_frame = ttk.Frame(main_frame)
        file_frame.grid(row=0, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        
        ttk.Button(file_frame, text="Load Source", command=self.load_source).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(file_frame, text="Save Source", command=self.save_source).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(file_frame, text="New File", command=self.new_file).pack(side=tk.LEFT, padx=(0, 5))
        
        # Assembly operations frame
        asm_frame = ttk.Frame(main_frame)
        asm_frame.grid(row=0, column=2, sticky=(tk.E,), pady=(0, 10))
        
        ttk.Button(asm_frame, text="Assemble", command=self.assemble).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(asm_frame, text="Save Binary", command=self.save_binary).pack(side=tk.LEFT)
        
        # Source code editor
        ttk.Label(main_frame, text="Source Code:").grid(row=1, column=0, sticky=tk.W)
        
        self.source_text = scrolledtext.ScrolledText(main_frame, width=60, height=15)
        self.source_text.grid(row=2, column=0, columnspan=3, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 10))
        
        # Output area
        ttk.Label(main_frame, text="Output:").grid(row=3, column=0, sticky=tk.W)
        
        self.output_text = scrolledtext.ScrolledText(main_frame, width=60, height=8, state=tk.DISABLED)
        self.output_text.grid(row=4, column=0, columnspan=3, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Status bar
        self.status_var = tk.StringVar()
        self.status_var.set("Ready")
        status_bar = ttk.Label(main_frame, textvariable=self.status_var, relief=tk.SUNKEN)
        status_bar.grid(row=5, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(10, 0))
        
        # Load sample if it exists
        self.load_sample_if_exists()
        
    def load_sample_if_exists(self):
        """Load sample.asm if it exists."""
        if os.path.exists("sample.asm"):
            try:
                with open("sample.asm", 'r') as f:
                    content = f.read()
                self.source_text.delete(1.0, tk.END)
                self.source_text.insert(1.0, content)
                self.status_var.set("Loaded sample.asm")
            except Exception as e:
                self.status_var.set(f"Error loading sample: {e}")
    
    def load_source(self):
        """Load assembly source file."""
        filename = filedialog.askopenfilename(
            title="Select source file",
            filetypes=[("Assembly files", "*.asm"), ("All files", "*.*")]
        )
        if filename:
            try:
                with open(filename, 'r') as f:
                    content = f.read()
                self.source_text.delete(1.0, tk.END)
                self.source_text.insert(1.0, content)
                self.current_file = filename
                self.status_var.set(f"Loaded: {filename}")
            except Exception as e:
                messagebox.showerror("Error", f"Could not load file: {e}")
    
    def save_source(self):
        """Save assembly source file."""
        if self.current_file:
            filename = self.current_file
        else:
            filename = filedialog.asksaveasfilename(
                title="Save source file",
                defaultextension=".asm",
                filetypes=[("Assembly files", "*.asm"), ("All files", "*.*")]
            )
            if not filename:
                return
        
        try:
            content = self.source_text.get(1.0, tk.END)
            with open(filename, 'w') as f:
                f.write(content)
            self.current_file = filename
            self.status_var.set(f"Saved: {filename}")
        except Exception as e:
            messagebox.showerror("Error", f"Could not save file: {e}")
    
    def new_file(self):
        """Create new file."""
        self.source_text.delete(1.0, tk.END)
        self.current_file = None
        self.status_var.set("New file")
    
    def assemble(self):
        """Assemble the current source code."""
        source_content = self.source_text.get(1.0, tk.END)
        
        if not source_content.strip():
            messagebox.showwarning("Warning", "No source code to assemble")
            return
        
        self.status_var.set("Assembling...")
        self.root.update()
        
        success, messages = self.assembler.assemble_from_text(source_content)
        
        # Update output
        self.output_text.config(state=tk.NORMAL)
        self.output_text.delete(1.0, tk.END)
        self.output_text.insert(1.0, messages)
        self.output_text.config(state=tk.DISABLED)
        
        if success:
            self.status_var.set("Assembly successful")
            messagebox.showinfo("Success", "Assembly completed successfully")
        else:
            self.status_var.set("Assembly failed - check output")
            messagebox.showerror("Error", "Assembly failed - check output for details")
    
    def save_binary(self):
        """Save binary output."""
        if not self.assembler.binary_code:
            messagebox.showwarning("Warning", "No binary code to save. Assemble first.")
            return
        
        filename = filedialog.asksaveasfilename(
            title="Save binary file",
            defaultextension=".bin",
            filetypes=[("Binary files", "*.bin"), ("All files", "*.*")]
        )
        if filename:
            success, messages = self.assembler.assemble_from_text(
                self.source_text.get(1.0, tk.END), 
                filename
            )
            
            # Update output with any new messages
            self.output_text.config(state=tk.NORMAL)
            self.output_text.delete(1.0, tk.END)
            self.output_text.insert(1.0, messages)
            self.output_text.config(state=tk.DISABLED)
            
            if success:
                self.status_var.set(f"Binary saved: {filename}")
                messagebox.showinfo("Success", f"Binary saved to {filename}")
            else:
                self.status_var.set("Save failed - check output")
                messagebox.showerror("Error", "Save failed - check output for details")


def main():
    root = tk.Tk()
    app = AssemblerGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
