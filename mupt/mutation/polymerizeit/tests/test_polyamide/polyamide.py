import argparse

import polymerizeit as pi
from gmx_simple import GMXsimple

def create_parser():
	parser = argparse.ArgumentParser(prog = 'polyamide-kolev-2014', usage = '%(prog)s [-h for help]', \
                                             description='Generate input files with unique atom IDs')
	parser.add_argument('-preprocess_database', '--preprocess_database', help = 'Input self.preprocess_database file name file', default="init.db")
	parser.add_argument('-main_database', '--main_database', help = 'Input self.preprocess_database file name file', default="main.db")
	parser.add_argument('-inputs_file',"--inputs_file", help="Parameters needed for polymerization")
	parser.add_argument('-iftop', help = 'Input .top/.itp file (Required).')
	parser.add_argument('-ifgro', help = 'Input .gro file (Required).')
	parser.add_argument('-oftop', "--oftop", help = 'Output .top/.itp file. \
                        Generated only when there is a reaction.', default = "rxn.top")
	parser.add_argument('-ofgro', "--ofgro", help = 'Output .gro file. \
                        Generated only when there is a reaction.', default = "rxn.gro")
	parser.add_argument('-nangdih', "--nangdih", help = "Define new angles and dihedrals required \
                         after amide bond formation.", default = bool(True))
	parser.add_argument('-ff', "--ff", help = "Path to force field files.", default = "amberff99.ff")
	parser.add_argument('-db', "--db", help = "Path to sqlite database.", default = "sqlite_memdb.db")
	parser.add_argument('-log', "--log", help = "Enabling this flag creates a log file.", default = "yes")
	parser.add_argument('-logfile', "--logfile", help = "Log filename.", default = "polymerizeit.log")
	parser.add_argument('-iter', "--iter", help = "minimization-equilibration cycle number.")

	return parser

# Standard boilerplate to call the main() function to begin the program.
if __name__=="__main__":
    try:
		
        args=create_parser().parse_args()
        param_args=vars(args)
        run=GMXsimple(**param_args)
        r_val = run.run(0,10)
    except (RuntimeError, TypeError, NameError) as e:
       print(e)
       print("Something went wrong in checking the reactions")	
