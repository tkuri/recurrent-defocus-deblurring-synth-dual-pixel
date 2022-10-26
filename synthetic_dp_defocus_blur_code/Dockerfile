FROM cuda11.1_pytorch1.8

RUN apt-get update && apt-get upgrade -y \
 && apt-get install -y \
    libmagickwand-dev
RUN pip install --upgrade pip \
 && pip install wand