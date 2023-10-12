# Language Circuits
 
Notebooks generally need to be moved to the root level directory to work.


### Notebooks

Some notebooks include the line `sys.path.append('../')` to add the root directory to the system path. This enables importing modules from sibling directories. From PEP 328:

> Relative imports use a module's __name__ attribute to determine that module's position in the package hierarchy. 
If the module's name does not contain any package information (e.g. it is set to '__main__' [as in all notebooks]) then relative imports are 
resolved as if the module were a top level module, regardless of where the module is actually located on the file system.