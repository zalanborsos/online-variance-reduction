FROM python:3.6
ADD vrb /vrb
ADD setup.py / 
RUN pip install numpy nose Cython
RUN python setup.py build_ext --inplace
RUN nosetests