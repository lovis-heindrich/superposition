# On local machine, copy file from directory to remote machine
scp -P <port_number> -r /Users/path/to/local_directory root@<remote_machine_ip_address>:/root/path/to/remote_directory

# Get IP address (Mac)
ipconfig getifaddr en0

# Get username (Mac)
whoami

# Get an available socket
python -c 'import socket; s=socket.socket(); s.bind(("", 0)); print(s.getsockname()[1]); s.close()'

# On remote machine, request file copy from local machine
sudo apt-get install openssh-server
scp username@local_host_ip_address:/Users/path/to/local_directory root/path/to/remote_directory/