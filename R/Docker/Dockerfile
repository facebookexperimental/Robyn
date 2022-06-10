# building docker image to use R-Studio in a browser

FROM rocker/verse:4.1.0

# WE EXPORT PATH FOR CONDA
ENV PATH="/opt/conda/bin:${PATH}"

# UPDATE A SERIES OF PACKAGES
RUN apt-get update --fix-missing && apt-get install -y ca-certificates libglib2.0-0 libxext6 libsm6 libxrender1 libxml2-dev

# INSTALL PYTHON 3 AND ANACONDA
RUN apt-get install -y python3-pip python3.9-dev && pip3 install virtualenv \
&& wget --quiet https://repo.anaconda.com/archive/Anaconda3-5.3.0-Linux-x86_64.sh -O ~/anaconda.sh \
&& /bin/bash ~/anaconda.sh -b -p /opt/conda && rm ~/anaconda.sh \
&& ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh \
&& echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc

# ACTIVATE CONDA ENVIRONMENT
RUN echo "source activate base" > ~/.bashrc

# WRITE RETICULATE_PYTHON VARIABLE IN .Renviron
RUN echo "RETICULATE_PYTHON = '/opt/conda/bin'" > .Renviron

## copy files
COPY /install_robyn.R /install_robyn.R

## install packages
RUN Rscript /install_robyn.R

COPY /demo_docker.R /home/guest/