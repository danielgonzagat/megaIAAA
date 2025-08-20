import subprocess, tempfile, textwrap, resource, os, signal, json

def run_mbpp(code: str, tests: str, timeout_s: int = 3) -> dict:
    """Runs Python code + tests in a constrained subprocess.
    Not a full container, but enforces CPU/mem caps for quick smoke checks.
    """
    with tempfile.TemporaryDirectory() as td:
        fn = os.path.join(td, "prog.py")
        test_fn = os.path.join(td, "test_prog.py")
        open(fn, "w").write(code)
        open(test_fn, "w").write(tests)
        cmd = ["python","-m","pytest","-q", test_fn]
        def preexec():
            # soft limits
            resource.setrlimit(resource.RLIMIT_CPU, (timeout_s, timeout_s))
            resource.setrlimit(resource.RLIMIT_AS, (512*1024*1024, 512*1024*1024))
        try:
            p = subprocess.run(cmd, cwd=td, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                               preexec_fn=preexec, timeout=timeout_s+2, text=True)
            ok = (p.returncode == 0)
            return {"ok": ok, "out": p.stdout}
        except Exception as e:
            return {"ok": False, "out": str(e)}
