all:
	cd operator_py/cython/; python3 setup.py build_ext --inplace; rm -rf build; cd ../../
clean:
	cd operator_py/cython/; rm *.so *.c *.cpp; cd ../../
