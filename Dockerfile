FROM tleyden5iwx/caffe-gpu-master
MAINTAINER Sean Chuang <sean_chuang@htc.com>

RUN apt-get install -y \
    python-imaging \
    tmux \
    vim

RUN pip install NearPy

# update caffe
RUN cd /opt/caffe && \
    git checkout Makefile && \
    git pull

RUN cd /opt/caffe && \
    cp Makefile.config.example Makefile.config && \
    echo "CPU_ONLY := 1" >> Makefile.config && \
    echo "CXX := /usr/bin/g++-4.6" >> Makefile.config && \
    sed -i 's/CXX :=/CXX ?=/' Makefile && \
    make all

RUN cd /opt/caffe && make pycaffe

# install zmq
RUN cd /opt && \
    wget http://download.zeromq.org/zeromq-3.2.5.tar.gz && \
    tar -zxvf zeromq-3.2.5.tar.gz && \
    cd zeromq-3.2.5 && \
    ./configure --without-libsodium && \
    make && make install    
ADD files/zmq/zmq.hpp /usr/local/include/zmq.hpp
RUN pip install --no-use-wheel pyzmq

# install dlib
RUN cd /opt && \
    git clone https://github.com/davisking/dlib.git

RUN cd /opt/dlib && \
    sed -i 's/long size = 200/long size = 100/' dlib/image_transforms/interpolation.h && \
    sed -i 's/long size = 200/long size = 100/' dlib/image_transforms/interpolation_abstract.h

# Add file to example
ADD files/dlib/face_landmark_detection_service.cpp /opt/dlib/examples/
ADD files/dlib/CMakeLists.txt /opt/dlib/examples/

RUN cd /opt/dlib/dlib && \
    echo "#define DLIB_JPEG_SUPPORT" >> config.h && \ 
    echo "#define DLIB_PNG_SUPPORT" >> config.h && \ 
    echo "#define DLIB_USE_FFTW" >> config.h && \ 
    echo "#define DLIB_USE_BLAS" >> config.h

RUN cd /opt/dlib/examples && \
    mkdir build && \
    cd build && \
    cmake .. -DUSE_SSE4_INSTRUCTIONS=ON && \
    sed -i '1 s/$/ -lzmq/' CMakeFiles/face_landmark_detection_service.dir/link.txt && \
    cmake --build . --config Release

RUN ldconfig
# add face-detect folder
ADD face-detect /opt/face-detect
RUN cp /opt/dlib/examples/build/face_landmark_detection_service /opt/face-detect/

# add flask-server folder
RUN pip install Flask pymongo simplejson 
#ADD flask-server /opt/flask-server


# run server
# EXPOSE 8888
# WORKDIR /opt/face-detect
# CMD ["run.sh"]

# WORKDIR /opt/flask-server
# CMD ["python", "server.py"]

# cleanup
RUN apt-get clean && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

