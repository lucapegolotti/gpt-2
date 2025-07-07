import os


class LogManager:
    def __init__(self, dir="log"):
        os.makedirs(dir, exist_ok=True)
        self.log_file = os.path.join(dir, f"log.txt")
        with open(self.log_file, "w") as f:
            pass

    def to_file(self, step, label, value):
        with open(self.log_file, "a") as f:
            f.write(f"{step} {label} {value:.6f}\n")
