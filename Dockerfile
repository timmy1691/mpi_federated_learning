FROM  python:3.10.13

# set the working directory
WORKDIR /scripts

COPY . /scripts

# install dependencies

ENV MPI_DIR=/opt/ompi
ENV PATH="$MPI_DIR/bin:$HOME/.local/bin:$PATH"
ENV LD_LIBRARY_PATH="$MPI_DIR/lib:$LD_LIBRARY_PATH"

# WORKDIR $HOME

# RUN apt-get -q update \
#     && apt-get install -y \
#     gcc gfortran binutils \
#     # && pip3 install --upgrade pip \
#     && apt-get clean \
#     && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

ADD https://download.open-mpi.org/release/open-mpi/v3.1/openmpi-3.1.4.tar.bz2 .
RUN tar xf openmpi-3.1.4.tar.bz2 \
    && cd openmpi-3.1.4 \
    && ./configure --prefix=$MPI_DIR \
    && make -j4 all \
    && make install \
    && cd .. && rm -rf \
    openmpi-3.1.4 openmpi-3.1.4.tar.bz2 /tmp/*

RUN pip3 install --user -U setuptools \
    && pip3 install --user mpi4py

# COPY ./actual_req.txt ./requirements.txt
# RUN apk update && apk add python3-dev \
#                         gcc \
#                         libc-dev
RUN pip install --no-cache-dir -r requirements.txt

RUN python3 -m pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116
RUN pip3 install torch --index-url https://download.pytorch.org/whl/cu118

# Create a non-root user 'mpiuser' and switch to it
# RUN adduser --disabled-password --gecos '' mpiuser
# USER mpiuser
# copy the src to the folder
# COPY ./src ./src
