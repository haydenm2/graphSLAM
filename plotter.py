import argparse
import matplotlib.pyplot as plt
import pdb

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Reads .g2o SLAM files and plots the results. Files must ")
    parser.add_argument('pre_file', 
        help="the .g20 pre_file representing the graph before optimization")
    parser.add_argument('post_file', 
        help="the .g20 pre_file representing the graph after optimization")

    files = []
    files.append(parser.parse_args().pre_file)
    files.append(parser.parse_args().post_file)

    node = {}   # maps the node ID to the node's (x,y) coordinate

    ax1 = plt.subplot(121)
    ax1.set_title("Before Optimization")
    ax2 = plt.subplot(122)
    ax2.set_title("After Optimization")

    for i,f in enumerate(files):
        curr_ax = ax1 if i == 0 else ax2

        with open(f) as fp:
            for line in fp:
                line = line.split()
                
                # save all poses
                if "VERTEX" in line[0] and "SE2" in line[0]:
                    v_id = line[1]
                    x = float(line[2])
                    y = float(line[3])

                    node[v_id] = (x,y)

                # save links between poses
                elif "EDGE" in line[0] and "POINT" not in line[0]:
                    v_out = node[line[1]]
                    v_in = node[line[2]]

                    x_vals = [v_out[0], v_in[0]]
                    y_vals = [v_out[1], v_in[1]]
                    curr_ax.plot(x_vals, y_vals, 'r')

    plt.show()