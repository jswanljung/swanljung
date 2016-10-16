import subprocess as sp
from multiprocessing import Pool
import os.path


def _tmake(target):
    target.make()


class Job(object):
    def __init__(self, params, targets=None):
        self.params = params
        self.targets = []
        if targets:
            self.targets.extend(targets)
        self.setup()

    def setup(self):
        """ Abstract class called during intialization."""
        pass

    def add_target(self, target):
        self.targets.append(target)

    def add_targets(self, targets):
        self.targets.extend(targets)

    def make_targets(self, processes=1, force=False, dryrun=False):
        p = Pool(processes, maxtasksperchild=1)
        p.map(_tmake, self.targets)
        p.close()

    @property
    def name(self):
        if 'name' in self.params:
            return self.params['name']
        else:
            return None

    @property
    def description(self):
        if 'description' in self.params:
            return self.params['description']
        else:
            return None


class Target(object):

    def __init__(self, filename, command):
        """ Takes a filename (or list) and command (list of command+args)"""
        if isinstance(filename, str) or isinstance(filename, unicode):
            self.filename = [filename]
        else:
            self.filename = filename
        self.command = command

    def make(self, force=False, dryrun=False):
        all_exist = all([os.path.exists(f) for f in self.filename])
        if force or not all_exist:
            command = self.command
            c = 0
            if not dryrun:
                c = sp.call(command)
                runstr = ""
                if c:
                    runstr += "WARNING: {} exited with exit code {}.\n".format(" ".join(command), c)
                else:
                    runstr += "{} completed with exit code {}.\n".format(" ".join(command), c)
                for f in self.filename:
                    if os.path.exists(f):
                        runstr += "Successfully created {}.\n".format(f)
                    else:
                        runstr += "WARNING: {} was not created.\n".format(f)
                print(runstr)
            else:
                print("{} (dry run)".format(" ".join(command)))
        else:
            for f in self.filename:
                print("{} already exists, skipping.").format(f)
