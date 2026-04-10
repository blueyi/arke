# schedule_ir.py

class ScheduleIR:
    def __init__(self):
        # Minimal skeleton for ScheduleIR
        self.ops = []  # List of scheduled operations
        self.dependencies = {}  # Op dependencies for scheduling

    def add_op(self, op):
        self.ops.append(op)

    def add_dependency(self, src_op, dst_op):
        if dst_op not in self.dependencies:
            self.dependencies[dst_op] = []
        self.dependencies[dst_op].append(src_op)

    def __repr__(self):
        return f"ScheduleIR(ops={self.ops}, dependencies={self.dependencies})"
