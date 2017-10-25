#!/usr/bin/python

from env import TestEnv
from conf import LisaLogging

LisaLogging.setup()

SD_FLAGS = [
    ("SD_LOAD_BALANCE",        0x0001),
    ("SD_BALANCE_NEWIDLE",     0x0002),
    ("SD_BALANCE_EXEC",        0x0004),
    ("SD_BALANCE_FORK",        0x0008),
    ("SD_BALANCE_WAKE",        0x0010),
    ("SD_WAKE_AFFINE",         0x0020),
    ("SD_ASYM_CPUCAPACITY",    0x0040),
    ("SD_SHARE_CPUCAPACITY",   0x0080),
    ("SD_SHARE_POWERDOMAIN",   0x0100),
    ("SD_SHARE_PKG_RESOURCES", 0x0200),
    ("SD_SERIALIZE",           0x0400),
    ("SD_ASYM_PACKING",        0x0800),
    ("SD_PREFER_SIBLING",      0x1000),
    ("SD_OVERLAP",             0x2000),
    ("SD_NUMA",                0x4000),
]

def main():
    te = TestEnv()

    def iter_cpu_sd_flags(cpu):
        """
        Get the flags for a given CPU's sched_domains

        :param cpu: Logical CPU number whose sched_domains' flags we want
        :returns: Iterator over the flags, as an int, of each of that CPU's
                  domains, highest-level (i.e. typically "DIE") first.
        """
        base_path = '/proc/sys/kernel/sched_domain/cpu{}/'.format(cpu)
        for domain in sorted(te.target.list_directory(base_path)):
            flags_path = te.target.path.join(base_path, domain, 'flags')
            yield te.target.read_int(flags_path)

    for cpu in range(te.target.number_of_cpus):
        print "CPU{}".format(cpu)

        for domain, sd_flags in enumerate(iter_cpu_sd_flags(cpu)):
            print "  domain{}".format(domain)
            for flag_name, flag_val in SD_FLAGS:
                if sd_flags & flag_val:
                    print "    " + flag_name
            print

if __name__ == '__main__':
    main()
