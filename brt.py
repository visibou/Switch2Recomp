import sys; sys.path.insert(0,'.'); import switchrecomp as s
s.build_runtime('gcc', sys.argv[1], False, 'x11', print)
