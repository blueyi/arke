# instruction_ir.py

class InstructionIR:
    def __init__(self):
        # Minimal skeleton for InstructionIR
        self.instructions = []  # List of low-level instructions

    def add_instruction(self, instruction):
        self.instructions.append(instruction)

    def __repr__(self):
        return f"InstructionIR(instructions={self.instructions})"
