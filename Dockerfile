FROM spellrun/torch-cpu
RUN apt update && apt install -y ffmpeg
RUN luarocks install torch 
RUN luarocks install nn
RUN luarocks install image
RUN luarocks install lua-cjson
RUN luarocks install ffmpeg
RUN luarocks install qttorch
RUN luarocks install nnx

WORKDIR /src
COPY . .
