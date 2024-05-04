#!/bin/bash

# Remove the current saved plot
rm /home/Nischayee/hzz_analysis/worker_hzz/data/graph.png

# Record the start time
start_time=$(date +%s)

rabbitmq-plugins enable rabbitmq_management

# Start RabbitMQ server
service rabbitmq-server start

# Function to build counter container
build_counter() {
    docker build -t counter-image /home/Nischayee/hzz_analysis/counter-hzz/
    docker run --name counter_container -v "/home/Nischayee/hzz_analysis/worker_hzz:/app/data" counter-image python hzz_counter.py --number_workers "$1"
}

# Function to build worker containers
build_worker() {
    docker build -t worker-image /home/Nischayee/hzz_analysis/worker-hzz/
    for ((i = 0; i < $1+1; i++)); do
        docker run -d --name worker_container_$i -v "/home/Nischayee/hzz_analysis/worker_hzz:/app/data" worker-image python hzz_script.py --rank "$i"
    done
}

# Function to build collector container
build_collector() {
    docker build -t collector-image /home/Nischayee/hzz_analysis/collector-hzz/
    docker run -d --name collector_container -v "/home/Nischayee/hzz_analysis/worker_hzz:/app/data" collector-image python hzz_collector.py
}

# Main function
main() {
    # Check if the number of arguments is correct
    if [ "$#" -ne 1 ]; then
        echo "Usage: $0 <number_of_workers>"
        exit 1
    fi

    # Build counter container
    build_counter "$1"

    # Build worker containers
    build_worker "$(( $1 - 1 ))"

    # Build collector container
    build_collector "$1"
}

# Run the main function with provided argument
main "$@"

# Record end time when plot is actually produced
while [ ! -f /home/Nischayee/hzz_analysis/worker_hzz/data/graph.png ]; do
    sleep 1
done
end_time=$(date +%s)

duration=$((end_time - start_time))
echo "Total time taken for analysis: $duration seconds."
