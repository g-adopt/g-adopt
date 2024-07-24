all: params.log

params.log: gplates_global.py
	echo "running $<" >&2
	/usr/bin/time --format="$@ finished in %E" tsp -f python3 $<

clean:
	rm -f output* *.h5 params.log
