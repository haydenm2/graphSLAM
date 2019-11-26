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

    g = 128 / 255
    gray = (g,g,g)
    f, axes = plt.subplots(nrows=1, ncols=2)

    for i,f in enumerate(files):
        curr_ax = axes[i]
        curr_ax.set_facecolor(gray)

        vertices = {}  # maps the vertexs ID to the vertex's (x,y) coordinate

        with open(f) as fp:
            for line in fp:
                line = line.split()
                
                # save all vertices
                if "VERTEX" in line[0]:
                    v_id = line[1]
                    x = float(line[2])
                    y = float(line[3])

                    vertices[v_id] = (x,y)
                # save links between vertices
                elif "EDGE" in line[0]:
                    zorder = 1
                    color = 'r'

                    v_out = vertices[line[1]]
                    v_in = vertices[line[2]]

                    if "POINT" in line[0]:
                        zorder = 0
                        color = 'w'

                    x_vals = [v_out[0], v_in[0]]
                    y_vals = [v_out[1], v_in[1]]
                    curr_ax.plot(x_vals, y_vals, color=color, zorder=zorder)

    plt.show()