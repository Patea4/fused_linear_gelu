import torch
from torch.testing._internal.common_utils import TestCase
from torch.testing._internal.optests.generate_tests import opcheck
import unittest
import fused_linear_gelu

def reference_muladd(a, b, c):
    return a * b + c


class TestMyMulAdd(TestCase):
    def sample_inputs(self, device):
        def make_tensor(*size):
            return torch.randn(size, device=device, requires_grad=False)

        return [
            [make_tensor(3), make_tensor(3), 1],
            [make_tensor(20), make_tensor(20), 3.14],
            [make_tensor(20), make_tensor(20), -123],
            [make_tensor(2, 3), make_tensor(2, 3), -0.3],
        ]
    
    def _test_correctness(self, device):
        samples = self.sample_inputs(device)
        for args in samples:
            result = fused_linear_gelu.ops.mymuladd(*args)
            expected = reference_muladd(*args)
            torch.testing.assert_close(result, expected)

    @unittest.skipIf(not torch.cuda.is_available(), "requires cuda")
    def test_correctness_cuda(self):
        self._test_correctness("cuda")

    def _opcheck(self, device):
        # Use opcheck to check for incorrect usage of operator registration APIs
        samples = self.sample_inputs(device)
        for args in samples:
            opcheck(torch.ops.fused_linear_gelu.mymuladd, args)

    @unittest.skipIf(not torch.cuda.is_available(), "requires cuda")
    def test_opcheck_cuda(self):
        self._opcheck("cuda")
        

if __name__ == "__main__":
    unittest.main()
