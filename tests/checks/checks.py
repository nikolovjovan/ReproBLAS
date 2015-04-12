import copy
import itertools

import scripts.terminal as terminal
import tests.harness.harness as harness

class CheckSuite(harness.Suite):

  def __init__(self):
    self.checks = []
    self.check_rows = []
    self.args = []
    self.params = []

  def add_checks(self, checks, params, ranges):
    for args in itertools.product(*ranges):
      check_row = copy.deepcopy(checks)
      self.check_rows.append(check_row)
      self.checks += check_row
      self.params.append(params)
      self.args.append(args)

  def setup(self, **kwargs):
    for check_row, params, args in zip(self.check_rows, self.params, self.args):
      for check in check_row:
        check.setup(flags = terminal.flags(params, args), **kwargs)

  def get_command_list(self):
    command_list = []
    for check in self.checks:
      command_list += check.get_command_list()
    return command_list

  def parse_output_list(self, output_list):
    for check in self.checks:
      check.parse_output_list(output_list[:len(check.get_command_list())])
      output_list = output_list[len(check.get_command_list()):]

  def get_header(self):
    return ["Check", "Res"]

  def get_align(self):
    return ["l", "c"]

  def get_dtype(self):
    return ["t", "t"]

  def get_cols_width(self, max_width):
    return [max_width - 1 - 1 - 4 - 1, 4]

  def get_rows(self):
    passed = 0
    failed = 0
    na = 0
    rows = []
    for check in self.checks:
      if check.get_result() == 0:
        rows.append([check.get_name(), "Pass"])
        passed += 1
      elif check.get_result() == 125:
        rows.append([check.get_output(), "N/A"])
        na += 1
      else:
        rows.append([check.get_output(), "Fail"])
        failed += 1
    emoticon = ":("
    if passed == len(self.checks):
      emoticon = ":D"
    rows.append(["Passed: {0}/{3} Failed: {1}/{3} N/A: {2}/{3}".format(passed, failed, na, len(self.checks)), emoticon])
    return rows

  def get_output(self):
    return "\n".join(self.get_rows)

  def get_result(self):
    return "\n".join(self.get_rows)

class CheckTest(harness.ExecutableTest):

  def get_name(self):
    """
    return the name of the test
    """
    return "{} {}".format(self.name, self.flags)

  def parse_output_list(self, output_list):
    """
    parse the output of the command set. The output will be given as a list of
    (return code, output)
    """
    assert len(output_list) == 1, "ReproBLAS error: unexpected test output"
    self.output = output_list[0][1]
    self.result = output_list[0][0]

  def get_output(self):
    """
    return all relevant output (mostly for debugging)
    """
    return self.output

  def get_result(self):
    """
    return test result
    """
    return self.result

class ValidateInternalUFPTest(CheckTest):
  executable = "tests/checks/validate_internal_ufp"
  name = "validate_internal_ufp"

class ValidateInternalUFPFTest(CheckTest):
  executable = "tests/checks/validate_internal_ufpf"
  name = "validate_internal_ufpf"

class ValidateInternalDAMAXTest(CheckTest):
  executable = "tests/checks/validate_internal_damax"
  name = "validate_internal_damax"

class ValidateInternalZAMAXTest(CheckTest):
  executable = "tests/checks/validate_internal_zamax"
  name = "validate_internal_zamax"

class ValidateInternalSAMAXTest(CheckTest):
  executable = "tests/checks/validate_internal_samax"
  name = "validate_internal_samax"

class ValidateInternalCAMAXTest(CheckTest):
  executable = "tests/checks/validate_internal_camax"
  name = "validate_internal_camax"

class ValidateInternalRDBLAS1Test(CheckTest):
  executable = "tests/checks/validate_internal_rdblas1"
  name = "validate_internal_rdblas1"

class ValidateInternalRZBLAS1Test(CheckTest):
  executable = "tests/checks/validate_internal_rzblas1"
  name = "validate_internal_rzblas1"

class ValidateInternalRSBLAS1Test(CheckTest):
  executable = "tests/checks/validate_internal_rsblas1"
  name = "validate_internal_rsblas1"

class ValidateInternalRCBLAS1Test(CheckTest):
  executable = "tests/checks/validate_internal_rcblas1"
  name = "validate_internal_rcblas1"

class ValidateExternalRDSUMTest(CheckTest):
  base_flags = "-w rdsum"
  executable = "tests/checks/validate_external_rdblas1"
  name = "validate_external_rdsum"

class ValidateExternalRDASUMTest(CheckTest):
  base_flags = "-w rdasum"
  executable = "tests/checks/validate_external_rdblas1"
  name = "validate_external_rdasum"

class ValidateExternalRDNRM2Test(CheckTest):
  base_flags = "-w rdnrm2"
  executable = "tests/checks/validate_external_rdblas1"
  name = "validate_external_rdnrm2"

class ValidateExternalRDDOTTest(CheckTest):
  base_flags = "-w rddot"
  executable = "tests/checks/validate_external_rdblas1"
  name = "validate_external_rddot"

class ValidateExternalRZSUMTest(CheckTest):
  base_flags = "-w rzsum"
  executable = "tests/checks/validate_external_rzblas1"
  name = "validate_external_rzsum"

class ValidateExternalRDZASUMTest(CheckTest):
  base_flags = "-w rdzasum"
  executable = "tests/checks/validate_external_rzblas1"
  name = "validate_external_rdzasum"

class ValidateExternalRDZNRM2Test(CheckTest):
  base_flags = "-w rdznrm2"
  executable = "tests/checks/validate_external_rzblas1"
  name = "validate_external_rdznrm2"

class ValidateExternalRZDOTUTest(CheckTest):
  base_flags = "-w rzdotu"
  executable = "tests/checks/validate_external_rzblas1"
  name = "validate_external_rzdotu"

class ValidateExternalRZDOTCTest(CheckTest):
  base_flags = "-w rzdotc"
  executable = "tests/checks/validate_external_rzblas1"
  name = "validate_external_rzdotc"

class ValidateExternalRSSUMTest(CheckTest):
  base_flags = "-w rssum"
  executable = "tests/checks/validate_external_rsblas1"
  name = "validate_external_rssum"

class ValidateExternalRSASUMTest(CheckTest):
  base_flags = "-w rsasum"
  executable = "tests/checks/validate_external_rsblas1"
  name = "validate_external_rsasum"

class ValidateExternalRSNRM2Test(CheckTest):
  base_flags = "-w rsnrm2"
  executable = "tests/checks/validate_external_rsblas1"
  name = "validate_external_rsnrm2"

class ValidateExternalRSDOTTest(CheckTest):
  base_flags = "-w rsdot"
  executable = "tests/checks/validate_external_rsblas1"
  name = "validate_external_rsdot"

class ValidateExternalRCSUMTest(CheckTest):
  base_flags = "-w rcsum"
  executable = "tests/checks/validate_external_rcblas1"
  name = "validate_external_rcsum"

class ValidateExternalRSCASUMTest(CheckTest):
  base_flags = "-w rscasum"
  executable = "tests/checks/validate_external_rcblas1"
  name = "validate_external_rscasum"

class ValidateExternalRSCNRM2Test(CheckTest):
  base_flags = "-w rscnrm2"
  executable = "tests/checks/validate_external_rcblas1"
  name = "validate_external_rscnrm2"

class ValidateExternalRCDOTUTest(CheckTest):
  base_flags = "-w rcdotu"
  executable = "tests/checks/validate_external_rcblas1"
  name = "validate_external_rcdotu"

class ValidateExternalRCDOTCTest(CheckTest):
  base_flags = "-w rcdotc"
  executable = "tests/checks/validate_external_rcblas1"
  name = "validate_external_rcdotc"

class VerifyRDSUMTest(CheckTest):
  base_flags = "-w rdsum"
  executable = "tests/checks/verify_rdblas1"
  name = "verify_rdsum"

class VerifyRDASUMTest(CheckTest):
  base_flags = "-w rdasum"
  executable = "tests/checks/verify_rdblas1"
  name = "verify_rdasum"

class VerifyRDNRM2Test(CheckTest):
  base_flags = "-w rdnrm2"
  executable = "tests/checks/verify_rdblas1"
  name = "verify_rdnrm2"

class VerifyRDDOTTest(CheckTest):
  base_flags = "-w rddot"
  executable = "tests/checks/verify_rdblas1"
  name = "verify_rddot"

class VerifyRZSUMTest(CheckTest):
  base_flags = "-w rzsum"
  executable = "tests/checks/verify_rzblas1"
  name = "verify_rzsum"

class VerifyRDZASUMTest(CheckTest):
  base_flags = "-w rdzasum"
  executable = "tests/checks/verify_rzblas1"
  name = "verify_rdzasum"

class VerifyRDZNRM2Test(CheckTest):
  base_flags = "-w rdznrm2"
  executable = "tests/checks/verify_rzblas1"
  name = "verify_rdznrm2"

class VerifyRZDOTUTest(CheckTest):
  base_flags = "-w rzdotu"
  executable = "tests/checks/verify_rzblas1"
  name = "verify_rzdotu"

class VerifyRZDOTCTest(CheckTest):
  base_flags = "-w rzdotc"
  executable = "tests/checks/verify_rzblas1"
  name = "verify_rzdotc"

class VerifyRSSUMTest(CheckTest):
  base_flags = "-w rssum"
  executable = "tests/checks/verify_rsblas1"
  name = "verify_rssum"

class VerifyRSASUMTest(CheckTest):
  base_flags = "-w rsasum"
  executable = "tests/checks/verify_rsblas1"
  name = "verify_rsasum"

class VerifyRSNRM2Test(CheckTest):
  base_flags = "-w rsnrm2"
  executable = "tests/checks/verify_rsblas1"
  name = "verify_rsnrm2"

class VerifyRSDOTTest(CheckTest):
  base_flags = "-w rsdot"
  executable = "tests/checks/verify_rsblas1"
  name = "verify_rsdot"

class VerifyRCSUMTest(CheckTest):
  base_flags = "-w rcsum"
  executable = "tests/checks/verify_rcblas1"
  name = "verify_rcsum"

class VerifyRSCASUMTest(CheckTest):
  base_flags = "-w rscasum"
  executable = "tests/checks/verify_rcblas1"
  name = "verify_rscasum"

class VerifyRSCNRM2Test(CheckTest):
  base_flags = "-w rscnrm2"
  executable = "tests/checks/verify_rcblas1"
  name = "verify_rscnrm2"

class VerifyRCDOTUTest(CheckTest):
  base_flags = "-w rcdotu"
  executable = "tests/checks/verify_rcblas1"
  name = "verify_rcdotu"

class VerifyRCDOTCTest(CheckTest):
  base_flags = "-w rcdotc"
  executable = "tests/checks/verify_rcblas1"
  name = "verify_rcdotc"

all_checks = {"validate_internal_ufp": ValidateInternalUFPTest,\
              "validate_internal_ufpf": ValidateInternalUFPFTest,\
              "validate_internal_damax": ValidateInternalDAMAXTest,\
              "validate_internal_zamax": ValidateInternalZAMAXTest,\
              "validate_internal_samax": ValidateInternalSAMAXTest,\
              "validate_internal_camax": ValidateInternalCAMAXTest,\
              "validate_internal_rdblas1": ValidateInternalRDBLAS1Test,\
              "validate_internal_rzblas1": ValidateInternalRZBLAS1Test,\
              "validate_internal_rsblas1": ValidateInternalRSBLAS1Test,\
              "validate_internal_rcblas1": ValidateInternalRCBLAS1Test,\
              "validate_external_rdsum": ValidateExternalRDSUMTest,\
              "validate_external_rdasum": ValidateExternalRDASUMTest,\
              "validate_external_rdnrm2": ValidateExternalRDNRM2Test,\
              "validate_external_rddot": ValidateExternalRDDOTTest,\
              "validate_external_rzsum": ValidateExternalRZSUMTest,\
              "validate_external_rdzasum": ValidateExternalRDZASUMTest,\
              "validate_external_rdznrm2": ValidateExternalRDZNRM2Test,\
              "validate_external_rzdotu": ValidateExternalRZDOTUTest,\
              "validate_external_rzdotc": ValidateExternalRZDOTCTest,\
              "validate_external_rssum": ValidateExternalRSSUMTest,\
              "validate_external_rsasum": ValidateExternalRSASUMTest,\
              "validate_external_rsnrm2": ValidateExternalRSNRM2Test,\
              "validate_external_rsdot": ValidateExternalRSDOTTest,\
              "validate_external_rcsum": ValidateExternalRCSUMTest,\
              "validate_external_rscasum": ValidateExternalRSCASUMTest,\
              "validate_external_rscnrm2": ValidateExternalRSCNRM2Test,\
              "validate_external_rcdotu": ValidateExternalRCDOTUTest,\
              "validate_external_rcdotc": ValidateExternalRCDOTCTest,\
              "verify_rdsum": VerifyRDSUMTest,\
              "verify_rdasum": VerifyRDASUMTest,\
              "verify_rdnrm2": VerifyRDNRM2Test,\
              "verify_rddot": VerifyRDDOTTest,\
              "verify_rzsum": VerifyRZSUMTest,\
              "verify_rdzasum": VerifyRDZASUMTest,\
              "verify_rdznrm2": VerifyRDZNRM2Test,\
              "verify_rzdotu": VerifyRZDOTUTest,\
              "verify_rzdotc": VerifyRZDOTCTest,\
              "verify_rssum": VerifyRSSUMTest,\
              "verify_rsasum": VerifyRSASUMTest,\
              "verify_rsnrm2": VerifyRSNRM2Test,\
              "verify_rsdot": VerifyRSDOTTest,\
              "verify_rcsum": VerifyRCSUMTest,\
              "verify_rscasum": VerifyRSCASUMTest,\
              "verify_rscnrm2": VerifyRSCNRM2Test,\
              "verify_rcdotu": VerifyRCDOTUTest,\
              "verify_rcdotc": VerifyRCDOTCTest}
