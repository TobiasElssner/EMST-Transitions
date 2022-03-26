#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
Implements a GUI to visualize the results. Based on the matplotlib example
https://matplotlib.org/3.1.0/gallery/user_interfaces/embedding_in_tk_sgskip.html

@author Tobias Elßner
"""


import tkinter
from tkinter import *
from matplotlib.backends.backend_tkagg import (
    FigureCanvasTkAgg, NavigationToolbar2Tk)
from matplotlib.backend_bases import key_press_handler
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from matplotlib import collections as mc
import numpy as np
from scipy.spatial.distance import euclidean as eu
from shapely.geometry import Point, Polygon, LineString, JOIN_STYLE, LinearRing
from Node import QueueNode, TreeNode
from PriorityQueue import PriorityQueue
from Subdivision import Subdivision
from PrimMST import PrimMST
from Boundary import Boundary
from Coordinate import Coordinate


"""
Gui to illustrate the project.
@author Tobias Elßner
"""


root = tkinter.Tk()
root.wm_title("GUI")

# Build a plot that fits the window
fig = plt.figure(root.winfo_width(), dpi=100)

# Create a subplot
ax = fig.add_subplot(111)

# Equal Ratio of x- and y-axis
ax.set_aspect('equal', 'box')

original_xmin, original_xmax = -10, 10
ax.set_xlim([original_xmin, original_xmax])

original_ymin, original_ymax = -10, 10
ax.set_ylim([original_ymin, original_ymax])

ax.set_title("Place points by mouse click.")

# Disable autoscale to prevent the coordinate from involuntary moving
plt.autoscale(enable=False, axis='both', tight=None)

# Zoom ratio for zooming out
zoom_ratio = 1.15

# Last zoom position
zoom_position = (original_xmin, original_xmax, original_ymin, original_ymax)

# Data structures for the MST
data_points = None

interim_msts = []
final_mst = None
mst_plot = None
additional_point = None

# Additional edge types needed to plot the wrong and optimal MST at the same time
common_edges = None
optimal_edges = None
erroneous_edges = None

# Allows faster plotting of correct_transitions
final_mst_edges = []
keys_to_edge_index = []

sub_division = None
correct_transitions = []
correct_transition_index = -1

erroneous_regions = []
erroneous_region_index = -1

# Line variables
starting_coordinate, end_coordinate = None, None
starting_point, end_point = None, None
line = None

extended_starting_coordinate, extended_end_coordinate = None, None
extended_starting_point, extended_end_point = None, None
extended_line = None

# Boolean variables to keep track where the User is
is_subdivision_computed = False

# Boolean to indicate where the user currently is
shows_correct_regions = False

is_line_drawn = False
is_extended = False

is_paused = False

# Pointer to the Help-Window
help_window = None

# Specify the window as master
canvas = FigureCanvasTkAgg(fig, master=root)
canvas.draw()
canvas.get_tk_widget().grid(row=0, column=0, columnspan=5, ipadx=10, ipady=10)

# Navigation toolbar
toolbarFrame = tkinter.Frame(master=root)
toolbarFrame.grid(row=1, column=0, columnspan=5)
toolbar = NavigationToolbar2Tk(canvas, toolbarFrame)


def on_key_press(event):
    """
    Handles all events during the interaction.
    :param event: tKinter event
    """
    key_press_handler(event, canvas, toolbar)


def onclick(event):
    """
    Handles click events on the canvas.
    :param event: tkinter (click) event
    """
    global is_subdivision_computed, sub_division

    # Boolean to check whether the coordinate system needs to be reframed
    # (Such that all points are on display)
    reframe_coordinate_system = True

    # Prevents handling clicks from outside the canvas
    if event.xdata is not None and event.ydata is not None:

        # While the MST is under construction, add the coordinates of clicks to the MST data points
        if not is_subdivision_computed:
            build_mst(event)
        else:
            # Otherwise, the points are meant to draw the line
            bottom_left_corner, top_right_corner = sub_division.box_corners[0], sub_division.box_corners[2]

            # If the line is complete, the coordinate system does not need to be adjusted
            if starting_coordinate is not None and end_coordinate is not None:
                reframe_coordinate_system = True

            # Make sure that the line does not exit the diagram
            if bottom_left_corner.x <= event.xdata <= top_right_corner.x and \
                    bottom_left_corner.y <= event.ydata <= top_right_corner.y:

                draw_line(event)
            else:
                plt.title("Place the line within the Diagram.")
                reframe_coordinate_system = False

        # Adjust the coordinate system if necessary
        if reframe_coordinate_system:

            center_mst()

            # Update the canvas
            canvas.draw()


def build_mst(event):
    """
    Build the MST.
    :param event: tkinter (click) event
    """
    global data_points, interim_msts, final_mst, final_mst_edges, keys_to_edge_index, mst_plot, ax

    # Initialize the data_points array if None
    if data_points is None:
        data_points = np.array([[event.xdata, event.ydata]])

    # Else, add the new point to the existing ones
    else:
        data_points = np.append(data_points, [[event.xdata, event.ydata]], axis=0)

    # Build the MST when more than one point is present
    if data_points.shape[0] > 1:
        ax.clear()
        ax.plot([data_points[i][0] for i in range(data_points.shape[0])],
                [data_points[i][1] for i in range(data_points.shape[0])],
                "ok", markersize=5)

        # Initialize a new matrix with the number of data points as number of rows and columns
        # By default, every entry contains -1
        keys_to_edge_index = np.full((data_points.shape[0], data_points.shape[0]), -1)

        tree = PrimMST(data_points)
        final_mst = tree
        final_mst_edges = []

        # For each node in the tree, plot the edges to its chilren
        for node in tree.mst_nodes:
            parent_point = data_points[node.key, :]

            for child in node.children:
                child_point = data_points[child.key, :]

                # Store the index of the edge between parent and child
                keys_to_edge_index[node.key][child.key] = len(final_mst_edges)
                keys_to_edge_index[child.key][node.key] = len(final_mst_edges)

                final_mst_edges.append(((parent_point[0], parent_point[1]), (child_point[0], child_point[1])))

        # Add the MST to interim results, such that after and 'undo' operation it can be again easily retrieved
        interim_msts.append((final_mst_edges, final_mst, final_mst.weight))

        line_segments = mc.LineCollection(final_mst_edges, colors="black", linewidths=2)

        # Plot the segments
        mst_plot = ax.add_collection(line_segments)

        ax.margins(0.5)
        # Show the tree weight as title
        # TODO
        plt.title("Tree Weight: " + str(np.round(final_mst.weight, 2)) + " cm")

    # If only one point is present, there is no need to construct an MST
    else:
        ax.clear()
        ax.plot([data_points[0][0]], [data_points[0][1]], "ok", markersize=5)

        # Initialize a new matrix with the number of data points as number of rows and columns
        # By default, every entry contains -1
        keys_to_edge_index = np.full((data_points.shape[0], data_points.shape[0]), -1)

        final_mst = None
        final_mst_edges = []
        weight = 0.0

        interim_msts.append((final_mst_edges, final_mst, weight))

        line_segments = mc.LineCollection(final_mst_edges, colors="black", linewidths=2)

        mst_plot = ax.add_collection(line_segments)

        ax.margins(0.5)
        ax.set_title("Tree Weight: " + str(np.round(weight, 2)) + " cm")


def zoom_out():
    """
    Zoom out by the global factor 1.15.
    """
    global zoom_position, zoom_ratio, ax

    # Use the former bounds of the coordinate system, such that it start at the last zoom-position.
    xmin, xmax, ymin, ymax = zoom_position

    # Stretch the bounds of the coordinate system by the zoom-factor
    xmin *= zoom_ratio
    xmax *= zoom_ratio

    ymin *= zoom_ratio
    ymax *= zoom_ratio

    ax.set_xbound([xmin, xmax])
    ax.set_ybound([ymin, ymax])

    # Store the zoom-position again in bounds
    zoom_position = (xmin, xmax, ymin, ymax)

    # Set the aspect ratio and prevent the figure from resizing
    # Comment out to preserve the original proportions
    ax.set_aspect(abs((xmax - xmin) / (ymin - ymax)) * 1.0)

    # Update the canvas
    canvas.draw()


def center_diagram():
    """
    Center the diagram in the coordinate system.
    """
    global zoom_position, sub_division, ax, size

    # If there is a diagram at all
    if sub_division is not None:
        bottom_left_corner, top_right_corner = sub_division.box_corners[0], sub_division.box_corners[2]

        xmin, xmax, ymin, ymax = bottom_left_corner.x, top_right_corner.x, bottom_left_corner.y, top_right_corner.y

        # Add a delta to put a little space between the diagram and the boundaries of the canvas
        delta_x = (xmax - xmin) / 5
        delta_y = (ymax - ymin) / 5
        xmin -= delta_x
        xmax += delta_x
        ymin -= delta_y
        ymax += delta_y

        ax.set_xbound([xmin, xmax])
        ax.set_ybound([ymin, ymax])

        # Set the aspect ratio and prevent the figure from resizing
        # Comment out to preserve the original proportions
        ax.set_aspect(abs((xmax - xmin) / (ymin - ymax)) * 1.0)

        zoom_position = (xmin, xmax, ymin, ymax)

    # If there is no diagram yet, simply adjust the coordinate system
    # Such that the focus is on the MST
    else:
        center_mst()

    # Update the canvas
    canvas.draw()


def center_mst():
    """
    Adjusts the coordinate system such that the MST is visible.
    """
    global data_points, starting_coordinate, end_coordinate, is_extended, additional_point, \
        original_xmin, original_xmax, original_ymin, original_ymax, zoom_position, ax, size

    if data_points is None:
        return

    # Find the minimal and maximal x- and y-values
    cur_xmin, cur_xmax = np.min(np.append(data_points[:, 0], original_xmin)), \
                         np.max(np.append(data_points[:, 0], original_xmax))
    cur_ymin, cur_ymax = np.min(np.append(data_points[:, 1], original_ymin)), \
                         np.max(np.append(data_points[:, 1], original_ymax))

    # Update those with the starting and end points of the line
    if not is_extended:
        if starting_coordinate is not None:
            cur_xmin, cur_xmax = np.min([starting_coordinate.x, cur_xmin]), np.max([starting_coordinate.x, cur_xmax])
            cur_ymin, cur_ymax = np.min([starting_coordinate.y, cur_ymin]), np.max([starting_coordinate.y, cur_ymax])

        if end_coordinate is not None:
            cur_xmin, cur_xmax = np.min([end_coordinate.x, cur_xmin]), np.max([end_coordinate.x, cur_xmax])
            cur_ymin, cur_ymax = np.min([end_coordinate.y, cur_ymin]), np.max([end_coordinate.y, cur_ymax])

    # And with any additional point added through a transition
    if additional_point is not None:
        cur_xmin, cur_xmax = np.min([additional_point[0].get_data()[0][0], cur_xmin]), \
                             np.max([additional_point[0].get_data()[0][0], cur_xmax])
        cur_ymin, cur_ymax = np.min([additional_point[0].get_data()[1][0], cur_ymin]), \
                             np.max([additional_point[0].get_data()[1][0], cur_ymax])

    # Add again a delta
    delta_x = (cur_xmax - cur_xmin) / 5
    delta_y = (cur_ymax - cur_ymin) / 5

    # Only reframe the coordinate system if a point lies outside the original frame boundaries
    if cur_xmin < original_xmin or cur_xmax > original_xmax:
        cur_xmin -= delta_x
        cur_xmax += delta_x

    if cur_ymin < original_ymin or cur_ymax > original_ymax:
        cur_ymin -= delta_y
        cur_ymax += delta_y

    # Overwrite the bounds
    ax.set_xbound([cur_xmin, cur_xmax])
    ax.set_ybound([cur_ymin, cur_ymax])

    # Set the aspect ratio and prevent the figure from resizing
    # Comment out to preserve the original proportions
    ax.set_aspect(abs((cur_xmax - cur_xmin) / (cur_ymin - cur_ymax)) * 1.0)

    # Disable autoscale to prevent the coordinate from involuntary moving
    plt.autoscale(enable=False, axis='both', tight=None)

    # Set the zoom-position to those bounds
    zoom_position = (cur_xmin, cur_xmax, cur_ymin, cur_ymax)

    # Update the canvas
    canvas.draw()


def show_regions_handler():
    """
    Handler to switch between the correct and erroneous regions.
    :return: None
    """
    global is_subdivision_computed, shows_correct_regions, compute_regions_button, draw_line_button

    # No region can be shown for None or one data point
    if data_points is None:
        plt.title("Place at least 2 points.two")
        is_subdivision_computed = False

        # Update the canvas
        canvas.draw()
        return

    if len(data_points) < 2:
        ax.set_title("Place at least 2 points.")

        # Update the canvas
        canvas.draw()
        return

    # If the correct regions are currently on display, show the erroneous regions
    # Set shows_correct_regions to false (as it is negated by the click on this button)
    # Disable the draw-line-button
    # Re-name the compute regions button
    # And finally show the false regions
    if shows_correct_regions:
        shows_correct_regions = False

        draw_line_button.configure(state="disabled")
        plot_next_button.configure(state="active")
        plot_prev_button.configure(state="active")

        compute_regions_button.configure(text="Show MST Regions")

        show_erroneous_regions()

    # If the false regions are on display:
    # Set shows_correct_regions to true
    # Re-configure undo to undo_line()
    # Enable the draw-line-button
    # Disable plot_* buttons for now, as no line is drawn yet
    # Re-name the compute regions button
    # And finally show the correct regions
    else:
        shows_correct_regions = True

        undo_button.configure(command=undo_line)

        draw_line_button.configure(state="active")

        plot_next_button.configure(state="disabled")
        plot_prev_button.configure(state="disabled")

        compute_regions_button.configure(text="Show False Regions")

        show_mst_regions()


def plot_next_handler():
    """
    Handler to switch between plotting the next correct transition and the next false region.
    """

    global shows_correct_regions

    if shows_correct_regions:
        plot_next_correct_transition()
    else:
        plot_next_erroneous_region()


def plot_prev_handler():
    """
    Handler to switch between plotting the previous correct transition and the next false region.
    """

    global shows_correct_regions

    if shows_correct_regions:
        plot_prev_correct_transition()
    else:
        plot_prev_erroneous_region()


def show_mst_regions():
    """
    Shows the correct regions in which the MST has the same topology.
    :return None
    """
    global compute_regions_button, is_subdivision_computed, data_points, sub_division, mst_plot, final_mst_edges,\
        starting_coordinate, end_coordinate, starting_point, end_point, line, undo_button

    # If the data_points are None ore one, no regions can be shown
    # (There would only be one)
    # Sanity check
    if data_points is None:
        plt.title("Place at least 2 points.")
        is_mst_computed = False

        # Update the canvas
        canvas.draw()
        return

    if len(data_points) < 2:
        plt.title("Place at least 2 points.")
        is_mst_computed = False

        # Update the canvas
        canvas.draw()
        return

    # If the subdivision has not been computed yet, do it now
    if sub_division is None:
        sub_division = Subdivision(data_points)

    # To achieve larger differences in colors, pretend twice as much regions and skip every second one
    cmap = get_cmap(len(sub_division.topology_to_regions.keys()) * 2)
    count = -1

    # Clear the canvas, in case the user has previously viewed the false regions and  now returning to the correct ones
    ax.clear()

    # Add the MST again
    line_segments = mc.LineCollection(final_mst_edges, colors="black", linewidths=2)
    mst_plot = ax.add_collection(line_segments)

    ax.plot([data_points[i][0] for i in range(data_points.shape[0])],
            [data_points[i][1] for i in range(data_points.shape[0])],
            "ok", markersize=5)

    # Go over all topologies and color them with a distinct color
    for topology in sub_division.topology_to_regions:

        count += 2

        # Sanity check
        # Differentiate between Polygon and MultiPolygon regions
        # Ideally, the regions are valid Polygons
        for geometry in sub_division.topology_to_regions[topology]:
            if geometry.geom_type == 'Polygon' and geometry.is_valid and not geometry.is_empty:

                xs, ys = geometry.exterior.xy

                ax.fill(xs, ys, alpha=0.5, c=cmap(count), ec='none')

                # Uncomment to mark borders of all regions gray
                # plt.plot(xs, ys, alpha=0.25, c="gray")

            elif geometry.geom_type == 'MultiPolygon' or geometry.geom_type == 'GeometryCollection':
                for sub_geometry in geometry.geoms:
                    if sub_geometry.geom_type == 'Polygon' and geometry.is_valid and not geometry.is_empty:

                        xs, ys = sub_geometry.exterior.xy

                        ax.fill(xs, ys, alpha=0.5, c=cmap(count), ec='none')

                        # Uncomment to mark borders of all regions gray
                        # plt.plot(xs, ys, alpha=0.25, c="gray")

    # Disable autoscale to prevent the coordinate from involuntary moving
    plt.autoscale(enable=False, axis='both', tight=None)

    # Instruct user to draw the line
    ax.set_title("Click 'Draw Line' when ready.")

    # Re-configure the undo button to undo_line()
    undo_button.configure(command=undo_line)

    # Disconnect the canvas from events
    # No more points can be added unless 'Draw Line' is clicked
    canvas.mpl_disconnect(cid)

    # Center on the diagram
    center_diagram()

    # Update the canvas
    canvas.draw()


def ready_to_draw():
    """
    Method enables user to draw a line after being called.
    """
    global compute_regions_button, is_subdivision_computed, data_points,\
        cid, starting_coordinate, end_coordinate, starting_point, end_point, line

    # No line can be drawn for None or one data point
    # Sanity check
    if data_points is None:
        plt.title("Place at least 2 points.")
        is_mst_computed = False

        # Update the canvas
        canvas.draw()
        return

    if len(data_points) < 2:
        ax.set_title("Place at least 2 points.")

        # Update the canvas
        canvas.draw()
        return

    is_subdivision_computed = True

    # Disable draw line
    # It has now no function and could distract the user if still being active
    draw_line_button.configure(state="disabled")

    # Connect the canvas again such that starting- and end-points of the line can be placed
    cid = fig.canvas.mpl_connect('button_press_event', onclick)

    # Instruct user
    ax.set_title("Place a starting point.")

    # Update the canvas
    canvas.draw()


def draw_line(event):
    """
    Draw the actual line to show transitions.
    :param event: tkinter (click) event
    :return: None
    """
    global starting_coordinate, end_coordinate, starting_point, end_point, line, is_subdivision_computed,\
        draw_line_button, plot_next_button, plot_prev_button

    # If there is None or one point, no line is drawn
    # As no transitions need to be calculated (there is only one transition)
    if data_points is None:
        plt.title("Place at least 2 points.")
        is_subdivision_computed = False

        # Update the canvas
        canvas.draw()
        return

    if len(data_points) < 2:
        ax.set_title("Place at least 2 points.")

        # Update the canvas
        canvas.draw()
        return

    # Adjust start and end coordinates, as well as title-instructions
    # Depending on which points are already set
    if starting_coordinate is None:
        starting_coordinate = Coordinate(event.xdata, event.ydata)
        starting_point = ax.plot(event.xdata, event.ydata, 'ko', mfc='none', markersize=5)
        ax.set_title("Place an end point.")

    elif starting_coordinate is not None and end_coordinate is None:
        end_coordinate = Coordinate(event.xdata, event.ydata)
        end_point = ax.plot(event.xdata, event.ydata, 'ko', mfc='none', markersize=5)

        x_coordinates = [starting_coordinate.x, end_coordinate.x]
        y_coordinates = [starting_coordinate.y, end_coordinate.y]
        line = ax.plot(x_coordinates, y_coordinates, 'ko', linestyle="--", mfc='none', markersize=5)

        # (Too) elaborate explanation for 'Extend Line', 'Plot Prev', and 'Plot Next', because this is already
        # Summarized in 'Help'
        # ax.set_title("Click 'Extend Line' to expand the line,\n" +
        #             "or 'Plot Prev' and 'Plot Next' to view MSTs along the line.")
        ax.set_title("")

        # Reconfigure draw_line_button to extend_line, when starting- and end-point is set
        draw_line_button.configure(text="Extend Line", command=extend_line, state="active")

        # Also enable plot_* buttons, because now transitions can be displayed
        plot_next_button.configure(state="active")
        plot_prev_button.configure(state="active")


def extend_line():
    """
    Extend the drawn line to the boundaries of the diagram.
    """
    global draw_line_button, sub_division, line, starting_coordinate, end_coordinate, starting_point, end_point, is_extended, \
        extended_starting_coordinate, extended_end_coordinate, extended_starting_point, extended_end_point, extended_line,\
        correct_transitions, correct_transition_index

    # Reset transitions
    correct_transitions = []
    correct_transition_index = -1

    # Disable the draw_line_button
    draw_line_button.configure(state="disabled")

    # Only extend if not already done so
    if not is_extended:
        if starting_coordinate is not None and end_coordinate is not None:

            # Shapely cannot handle infinite lines, therefore the extended intersections with the bounding box
            # Is calculated with the Boundary class
            forward_dir = Coordinate(end_coordinate.x - starting_coordinate.x,
                                     end_coordinate.y - starting_coordinate.y)

            backward_dir = Coordinate(starting_coordinate.x - end_coordinate.x,
                                      starting_coordinate.y - end_coordinate.y)

            forward_line = Boundary(starting_coordinate, forward_dir, None, None)

            backward_line = Boundary(starting_coordinate, backward_dir, None, None)

            starting_point = starting_point.pop(0)
            starting_point.remove()

            end_point = end_point.pop(0)
            end_point.remove()

            # Check which boundaries are intersected, in both directions
            for border in sub_division.bounding_box:

                intersects, intersection = backward_line.intersects_boundary(border)

                if intersects:
                    extended_starting_coordinate = intersection
                    extended_starting_point = ax.plot(intersection.x, intersection.y, 'ko', mfc='none', markersize=5)

                intersects, intersection = forward_line.intersects_boundary(border)

                if intersects:
                    extended_end_coordinate = Coordinate(intersection.x, intersection.y)
                    extended_end_point = ax.plot(intersection.x, intersection.y, 'ko', mfc='none', markersize=5)

            # Remove the previous line
            line = line.pop(0)
            line.remove()

            # Plot the new one
            x_coordinates = [extended_starting_coordinate.x, extended_end_coordinate.x]
            y_coordinates = [extended_starting_coordinate.y, extended_end_coordinate.y]
            extended_line = ax.plot(x_coordinates, y_coordinates, 'ko', linestyle="--", mfc='none', markersize=5)

            is_extended = True

            # For convenience, plot the first transition
            plot_next_correct_transition()

            center_mst()

            # Disable autoscale to prevent the coordinate from involuntary moving
            plt.autoscale(enable=False, axis='both', tight=None)

        else:
            # Sanity instruction:
            # Line can only be extended if there exists a starting- and an end-point
            plt.title("The line needs a start and end point")

        # Update the canvas
        canvas.draw()


def calculate_transitions():
    """
    Calculates MST transitions along the line.
    """
    global sub_division, starting_coordinate, end_coordinate, \
        extended_starting_coordinate, extended_end_coordinate, is_extended,\
        correct_transitions

    if starting_coordinate is not None and end_coordinate is not None:

        # Sorts the intersections of the drawn line with the regions from start_ to end_coordinate
        # The line is not likely to cross every region, however, it is assumed that each region can
        # Be entered by every other region, thus resulting in an upper bound of (number of regions)².
        # The MST is updated whenever its topology changes
        # i.e. every time a region with a subset differing from the current one is entered
        intersection_pq = PriorityQueue(sub_division.num_of_regions**2, comp=is_greater_singlegton, dist=eu)

        # Line-Polygon-intersection is easier handled in Shapely, therefore conversion from
        # Coordinate/ Boundary to Point/ LineString
        sc, ec = Point(starting_coordinate.x, starting_coordinate.y), Point(end_coordinate.x, end_coordinate.y)

        if is_extended:
            sc, ec = Point(extended_starting_coordinate.x, extended_starting_coordinate.y), \
                     Point(extended_end_coordinate.x, extended_end_coordinate.y)

        shapely_line = LineString([sc, ec])

        # Loop over all topologies and their regions
        for topology in sub_division.topology_to_regions:

            for geometry in sub_division.topology_to_regions[topology]:
                if geometry.geom_type == "Polygon" and geometry.is_valid and not geometry.is_empty:
                    if shapely_line.intersects(geometry):

                        intersection = shapely_line.intersection(geometry)

                        # Differentiatiation between LineString and MultiLineString objects is crucial to prevent errors
                        # If the same region is crossed multiple times
                        # In the queue, use the distance between starting point of the line and entrance into the region
                        # As key, and the topology as value
                        if intersection.geom_type == 'LineString':
                            in_going = list(intersection.coords)[0]
                            dist = sc.distance(Point(in_going))
                            in_q_node = QueueNode(dist, topology)
                            intersection_pq.insert(in_q_node)

                        if intersection.geom_type == 'MultiLineString':
                            list_of_intersections = list(intersection.geoms)

                            for partial_line in list_of_intersections:

                                in_going = list(partial_line.coords)[0]

                                dist = sc.distance(Point(in_going))
                                in_q_node = QueueNode(dist, topology)
                                intersection_pq.insert(in_q_node)

        node = intersection_pq.pop()
        dist = node.key
        topology = node.value

        # Indicates that the starting point is
        is_starting_point = True

        # Topologies along the line
        correct_transitions = [(dist, topology)]

        # Retrieve the sorted transitions from the queue
        while not intersection_pq.is_empty():
            cur_dist, cur_topology = correct_transitions[-1]
            node = intersection_pq.pop()

            # This is the point where a region with a new topology is entered
            if node.value != cur_topology:

                # Use the point in middle between entrance and exit of the region(s) with the same topology
                # Any point x in the cell is sufficient as representative for the whole cell, because
                # "[t]he geographic neighbor set is invariant over all x in the cell",
                # See Monma & Suri, Chapter 4, "Classification of Topologies"
                cur_dist += node.key
                cur_dist /= 2

                # Easiest way to calculate the point on the line with distance cur_dist to the start_coordinate in shapely:
                # Compute a circle around the starting point with radius cur_dist and intersect that with the line
                # The intersection returns the straight between the start_coordinate and the coordinate where the line
                # Leaves the circle. This second coordinate is the one in question.
                point = tuple(list(shapely_line.intersection(sc.buffer(cur_dist)).coords)[1])

                # In case the last point is the starting coordinate
                # Consider not the mid point between starting coordinate and boundary,
                # But the starting coordinate itself
                # This is done for a better visual appearance
                if is_starting_point:
                    point = tuple(list(sc.coords)[0])
                    is_starting_point = False

                correct_transitions[-1] = update_edges(cur_topology, point)

                # Add the new topology to the list
                correct_transitions.append((node.key, node.value))

        # If the whole line crosses more than one region
        if len(correct_transitions) > 1:
            # Also consider the last point on the line at the border of the bounding box
            # In case more than one region is passed, the last point is the end coordinate
            # And not the mid point between the last crossing and the end coordinate
            # This is done for a better visual appearance
            point = tuple(list(ec.coords)[0])
            ignore, cur_topology = correct_transitions[-1]
            correct_transitions[-1] = update_edges(cur_topology, point)

        # If the whole line does not leave one region, take the start and end point
        else:
            # Distance can be ignored
            ignore, cur_topology = correct_transitions[-1]

            # The list of transitions contains only two transitions for the starting- and end point of the line
            correct_transitions = [update_edges(cur_topology, tuple(list(sc.coords)[0])),
                                   update_edges(cur_topology, tuple(list(ec.coords)[0]))]


def plot_next_correct_transition():
    """
    Plots the next transition of the MST along the drawn line.
    """

    global data_points, final_mst, mst_plot, additional_point, correct_transitions, correct_transition_index, is_extended, ann
    num_of_transitions = len(correct_transitions)

    # If the number of transitions is zero, they have not been calculated yet
    if num_of_transitions == 0:
        calculate_transitions()

        num_of_transitions = len(correct_transitions)

    # Sanity check if there are transitions at all
    if num_of_transitions != 0:

        # Increase the transition index
        correct_transition_index += 1

        # Take the index modulo the number of transitions to start again at the starting point of the line
        correct_transition_index = correct_transition_index % num_of_transitions

        # Remove the current MST plot
        if mst_plot is not None:
            mst_plot.remove()
            mst_plot = None

        # Remove the additional point from the last transition, if there is one
        if additional_point is not None:
            additional_point = additional_point.pop(0)
            additional_point.remove()
            additional_point = None

        # Retrieve information about the current transition
        weight, edges, additional_point = correct_transitions[correct_transition_index]

        # Plot the lines and the new additional point
        additional_point = ax.plot(additional_point[0], additional_point[1], 'ko', mfc='none', markersize=4)

        ax.plot([data_points[i][0] for i in range(data_points.shape[0])],
                [data_points[i][1] for i in range(data_points.shape[0])],
                "ok", markersize=5)

        line_segments = mc.LineCollection(edges, colors="black", linewidths=2)

        # Overwrite the current MST plot such that it can be completely removed in the next transition
        mst_plot = ax.add_collection(line_segments)

        center_mst()

        # Return the weight of the new tree in the plot title
        plt.title("Tree Weight: " + str(np.round(weight, 2)) + " cm")

        # Update the canvas
        canvas.draw()


def plot_prev_correct_transition():
    """
    Plot the previous transition along the line.
    """
    global correct_transitions, correct_transition_index

    # Simply go two steps back and call plot_next_correct_transition(), which goes one step forward
    # If transitions have been calculated.
    if len(correct_transitions) > 0:
        correct_transition_index -= 2

    # If no transitions have been computed yet, the index is initially at -1
    # Therefore, only 1 is subtracted, which is then added again in plot_next
    # And the transitions start at the line's end point (index = -1)
    else:
        correct_transition_index -= 1

    plot_next_correct_transition()


def is_greater_singlegton(key1, key2):
    """
    Compares the float keys key1 and key2
    :param key1: Float
    :param key2: Float
    :return: key1 is greater than key2.
    Thus, smaller keys "swim", and larger ones "sink"; the result is a MIN-PQ.
    """

    return key1 > key2


def get_cmap(n, name='hsv'):
    """
    Returns a function that maps each index in 0, 1, ..., n-1 to a distinct RGB color;
    the keyword argument name must be a standard mpl colormap name.
    Based on:
    https://stackoverflow.com/questions/14720331/how-to-generate-random-colors-in-matplotlib

    :param n: Integer indicating the number of distinct colors
    :param name: matplotlib color map
    :return: matplotlib color map containing n distinct colors
    """
    """"""

    return plt.cm.get_cmap(name, n)


def show_erroneous_regions():
    """
    Displays errors in the subdivision.
    """
    global data_points, sub_division, mst_plot, mst_tree,  final_mst_edges, additional_point, erroneous_regions, \
        is_extended, starting_coordinate, end_coordinate, starting_point, end_point, \
        extended_starting_point, extended_end_point, extended_starting_coordinate, extended_end_coordinate,\
        line, extended_line, correct_transitions, correct_transition_index,\
        draw_line_button, undo_button, plot_prev_button, plot_next_button, cid

    # Configure the undo_button to remove the false regions
    undo_button.configure(command=undo_errors)
    erroneous_regions = []

    # If the data_points are None ore one, no regions can be shown
    # (There would only be one)
    # Sanity check
    if data_points is None:
        plt.title("Place at least 2 points.")
        is_mst_computed = False

        # Update the canvas
        canvas.draw()
        return

    if len(data_points) < 2:
        ax.set_title("Place at least 2 points.")

        # Update the canvas
        canvas.draw()
        return

    # Sanity check: prevent accessing a None-sub_division
    if sub_division is None or len(sub_division.data_points) != len(data_points):
        sub_division = Subdivision(data_points)

    # Delete Line
    is_extended = False
    starting_coordinate, end_coordinate = None, None
    starting_point, end_point = None, None
    extended_starting_coordinate, extended_end_coordinate = None, None
    extended_starting_point, extended_end_point = None, None
    line = None
    extended_line = None

    # Remove the additional point from the last transition, if existing
    if additional_point is not None:
        additional_point = additional_point.pop(0)
        additional_point.remove()
        additional_point = None

    # Reset draw_line_button to ready_to_draw()
    draw_line_button.configure(text="Draw Line", command=ready_to_draw, state="disabled")

    # Reset transitions
    correct_transitions = []
    correct_transition_index = -1

    # Clear to remove any possible subdivision from show_mst_regions()
    ax.clear()

    # Disconnect canvas
    canvas.mpl_disconnect(cid)

    # Add the MST again
    line_segments = mc.LineCollection(final_mst_edges, colors="black", linewidths=2)
    mst_plot = ax.add_collection(line_segments)

    ax.plot([data_points[i][0] for i in range(data_points.shape[0])],
            [data_points[i][1] for i in range(data_points.shape[0])],
            "ok", markersize=5)

    for i in range(len(data_points)):

        others = [j for j in range(len(data_points)) if j != i]

        vor = sub_division.get_voronoi_region(i, others)
        xs, ys = vor.exterior.xy
        plt.plot(xs, ys, alpha=0.25, c="black")

    # Loop over all correct regions
    for region, subset in sub_division.correct_regions:

        # Sanity check
        # Differentiate between Polygon and MultiPolygon regions
        # Ideally, all regions are valid Polygons
        if region.geom_type == 'Polygon' and region.is_valid and not region.is_empty:

            # Color correct regions blue
            xs, ys = region.exterior.xy
            ax.fill(xs, ys, alpha=0.5, c="blue", ec='none')

            # Uncomment to also mark borders of correct regions gray
            # plt.plot(xs, ys, alpha=0.25, c="gray")


        elif region.geom_type == 'MultiPolygon' or region.geom_type == 'GeometryCollection':
            for geometry in region.geoms:
                if geometry.geom_type == 'Polygon' and geometry.is_valid and not geometry.is_empty:

                    # Color correct regions blue
                    xs, ys = geometry.exterior.xy
                    ax.fill(xs, ys, alpha=0.5, c="blue", ec='none')

                    # Uncomment to mark borders of all regions gray
                    # plt.plot(xs, ys, alpha=0.25, c="gray")

    # Min-PriorityQueue to sort the centroid coordinates of the false regions from left to right and bottom to top
    # This prevents erratic jumps from one region to another
    min_pq = PriorityQueue(len(sub_division.erroneous_regions), comp=is_greater_tuple, dist=eu)

    # Loop over all false regions
    for region, erroneous_subset, correct_subset in sub_division.erroneous_regions:

        # Sanity check
        # Differentiate between Polygon and MultiPolygon regions
        # Ideally, all regions are valid Polygons
        if region.geom_type == 'Polygon' and region.is_valid and not region.is_empty:

            # Color erroneous regions red
            xs, ys = region.exterior.xy
            ax.fill(xs, ys, alpha=0.5, c="red", ec='none')
            plt.plot(xs, ys, alpha=0.25, c="gray")

            # Only compute the wrong and correct tree if not already done so
            # In that case, the number erroneous transitions is equal to the number of erroneous regions
            if len(erroneous_regions) != len(sub_division.erroneous_regions):

                # Comvert the centroid coordinates to tuples to be able to hash them later
                centroid = tuple(list(region.centroid.coords)[0])

                # The nodes in the queue have the centroids as key and the false and correct subset as value
                q_node = QueueNode(centroid, (erroneous_subset, correct_subset, region.area))
                min_pq.insert(q_node)

        elif region.geom_type == 'MultiPolygon' or region.geom_type == 'GeometryCollection':
            for geometry in region.geoms:
                if geometry.geom_type == 'Polygon' and geometry.is_valid and not geometry.is_empty:

                    # Color erroneous regions red
                    xs, ys = geometry.exterior.xy
                    ax.fill(xs, ys, alpha=0.5, c="red", ec='none')
                    plt.plot(xs, ys, alpha=0.25,c="gray")

                    # Only compute the wrong and correct tree if not already done so
                    # In that case, the number erroneous transitions is equal to the number of erroneous regions
                    if len(erroneous_regions) != len(sub_division.erroneous_regions):

                        # Comvert the centroid coordinates to tuples to be able to hash them later
                        centroid = tuple(list(geometry.centroid.coords)[0])

                        # The nodes in the queue have the centroids as key and the false and correct subset as value
                        q_node = QueueNode(centroid, (erroneous_subset, correct_subset, geometry.area))
                        min_pq.insert(q_node)

    # Retrieve the sorted false regions from the queue
    while not min_pq.is_empty():

        q_node = min_pq.pop()

        centroid = q_node.key
        erroneous_subset, correct_subset, area = q_node.value

        # Since the centroid is part of returned tuple from compute_wrong_and_correct_tree(...), its result can be
        # Simply appended to the erroneous_regions list
        erroneous_regions.append((compute_wrong_and_correct_tree(erroneous_subset, correct_subset, centroid), area))

    # Display the overall area of false regions
    ax.set_title("Area of false Regions: " + str(np.round(sub_division.area_of_erroneous_regions, 2)) + r" cm$^2$" + "\n" +
                 "Total Area: " + str(np.round(sub_division.total_area, 2)) + r" cm$^2$")

    # ax.set_title(
    #      "Fläche falscher Regionen: " + str(np.round(sub_division.area_of_erroneous_regions, 2)) + r" cm$^2$" + "\n" +
    #      "Komplette Fläche: " + str(np.round(sub_division.total_area, 2)) + r" cm$^2$")

    # If there are no false regions, disable the plot-buttons
    if sub_division.area_of_erroneous_regions == 0.0:

        plot_prev_button.configure(state="disabled")
        plot_next_button.configure(state="disabled")

    # Center the diagram
    center_diagram()

    # Disable the canvas
    canvas.mpl_disconnect(cid)

    # Update the canvas
    canvas.draw()


def compute_wrong_and_correct_tree(actual_subset, optimal_subset, centroid):
    """
    Computes both the wrong and the correct MST for a given actual topology, optimal topology, and an additional point.
    :param actual_subset: Tuple containing the wrong subset of data point indices
    :param optimal_subset: Tuple containing the correct subset of data point indices
    :param centroid: Tuple containing coordinates of the centroid of a false region
    :return: quadruple (actual_weight, ae, optimal_weight, oe, ce, centroid):
    float weight of the wrong tree, distinct edges (coded as tuples of points) of the wrong tree,
    float weight of the correct tree, distinct edges (coded as tuples of points) of the correct tree,
    edges (coded as tuples of points) present in both the wrong and correct tree,
    and the the centroid coordinates as tuple
    """

    # Calculate edges for both the wrong and the correct tree
    actual_weight, actual_edges, centroid = update_edges(actual_subset, centroid)
    optimal_weight, optimal_edges, centroid = update_edges(optimal_subset, centroid)

    # Compute the edges which are in both trees
    common_edges = set()

    # Converting the list of tuples into sets makes operations easier and more efficient
    actual_edges = set(list(actual_edges))
    optimal_edges = set(list(optimal_edges))

    # Sort out common edges both in the actual and optimal tree
    for actual_edge in actual_edges:
        if actual_edge in optimal_edges:
            common_edges.add(actual_edge)

    for common_edge in common_edges:
        actual_edges.remove(common_edge)
        optimal_edges.remove(common_edge)

    actual_edges = list(actual_edges)
    optimal_edges = list(optimal_edges)
    common_edges = list(common_edges)

    # Plot common edges black, optimal edges blue, and wrong edges red
    ce = mc.LineCollection(common_edges, colors="black", linestyle="-", linewidths=2)
    oe = mc.LineCollection(optimal_edges, colors="blue", linestyle="--", linewidths=2)
    ae = mc.LineCollection(actual_edges, colors="red", linestyle=":", linewidths=2)

    # Returning the centroid again as part of the tuple makes appending the results to the erroneous_regions list easy
    return actual_weight, ae, optimal_weight, oe, ce, centroid


def update_edges(topology, additional_point):
    """
    Updates the edges of the MST according to the given topology and the given additional point.
    :param topology: Tuple consisting of the integer indices of the data points which are connected to the point
    :param additional_point: Tuple of the coordinates of the additional point
    :return: Tuple (new_weight, edges, point)
             consisting of
             the weight of the new Tree,
             its edges (coded as tuples), and
             the additional point (coded as 2d- tuple)
    """

    global data_points, final_mst, final_mst_edges, keys_to_edge_index

    new_weight = final_mst.weight

    # The edges between the elements of the subset which get deleted
    deleted_edges = set()

    # Copy edges to avoid permanent changes of the original tree
    edges = final_mst_edges.copy()

    for elem in topology:

        # Add edges to the additional point from those data points whose indices are given in the topology-tuple
        edges.append((additional_point, tuple(data_points[elem])))

        # Add the euclidean distance between every element in the subset and the newly added node
        new_weight += eu(data_points[elem, :], additional_point)

        # Remove the longest edge in the cycle
        for other_elem in topology:
            if other_elem != elem:
                parent_key, child_key = final_mst.longest_common_edges[elem][other_elem][0]
                parent_point = tuple(data_points[parent_key, :])

                # Redirect the edge (parent, child) to (parent, parent)
                # Avoids deleting in O(n)
                edges[keys_to_edge_index[parent_key][child_key]] = (parent_point, parent_point)

                deleted_edges.add((parent_key, child_key))

    # Delete the (longest) edges which would form cycles in the tree
    for (u, v) in deleted_edges:
        new_weight -= eu(data_points[u, :], data_points[v, :])

    return new_weight, edges, additional_point


def plot_next_erroneous_region():
    """
    Plots the next erroneous region.
    """

    global additional_point, mst_plot, erroneous_edges, optimal_edges, common_edges,\
        erroneous_regions, erroneous_region_index, ax

    # If there are erroneous regions to plot
    if len(erroneous_regions) > 0:

        # Increase the transition index
        erroneous_region_index += 1

        # Take the index modulo the number of transitions to start again at the starting point of the line
        erroneous_region_index = erroneous_region_index % len(erroneous_regions)

        # Remove the MST plot, if existing
        if mst_plot is not None:
            mst_plot.remove()
            mst_plot = None

        # Remove the additional point from the last transition, if existing
        if additional_point is not None:
            additional_point = additional_point.pop(0)
            additional_point.remove()
            additional_point = None

        # Remove any erroneous edges, if existing
        if erroneous_edges is not None:
            erroneous_edges.remove()
            erroneous_edges = None

        # Remove any optimal edges
        if optimal_edges is not None:
            optimal_edges.remove()
            optimal_edges = None

        # Remove any common edges, if existing
        if common_edges is not None:
            common_edges.remove()
            common_edges = None

        # Retrieve the weights and edges of the wrong and correct MST for the current region
        (actual_weight, actual_edges, optimal_weight, optimal_edges, common_edges, additional_point), area = \
            erroneous_regions[erroneous_region_index]

        # Compute the difference between both trees and show it in the plot's title
        dif = actual_weight - optimal_weight
        ax.set_title("Difference: " + str(np.round(dif, 2)) + " cm\n" +
                     "Area: " + str(np.round(area, 2)) + r" cm$^2$")

        # ax.set_title("Differenz: " + str(np.round(dif, 2)) + " cm\n" +
        #              "Fläche: " + str(np.round(area, 2)) + r" cm$^2$")

        # Add the edges again to the plot
        erroneous_edges = ax.add_collection(actual_edges)
        optimal_edges =  ax.add_collection(optimal_edges)
        common_edges = ax.add_collection(common_edges)

        # Add the additional point, and the original MST nodes
        additional_point = ax.plot(additional_point[0], additional_point[1], 'ko', mfc='none', markersize=4)

        ax.plot([data_points[i][0] for i in range(data_points.shape[0])],
                [data_points[i][1] for i in range(data_points.shape[0])],
                "ok", markersize=5)

        # Reframe the coordinate system to the new tree
        center_mst()

        # Update the canvas
        canvas.draw()

    # Otherwise, compute erroneous regions
    else:
        show_erroneous_regions()


def plot_prev_erroneous_region():
    global erroneous_regions, erroneous_region_index

    # Simply go two steps back and call plot_next_erroneous_region(), which goes one step forward
    # If transitions have been calculated.
    if len(erroneous_regions) > 0:
        erroneous_region_index -= 2

    # If no transitions have been computed yet, the index is initially at -1
    # Therefore, only 1 is subtracted, which is then added again in plot_next
    # And the transitions start at the line's end point (index = -1)
    else:
        erroneous_region_index -= 1

    plot_next_erroneous_region()


def is_greater_tuple(key1, key2):
    """
    Compares the float keys key1 and key2
    :param key1: Float
    :param key2: Float
    :return: key1 is greater than key2.
    Thus, smaller keys "swim", and larger ones "sink"; the result is a MIN-PQ.
    """

    if key1[0] == key2[0]:
        return key1[1] > key2[1]

    return key1[0] > key2[0]


def undo_last_node():
    """
    Remove the last MST node.
    :return: None
    """
    global data_points, interim_msts, final_mst, final_mst_edges, sub_division, correct_transitions, \
        correct_transition_index, erroneous_regions, erroneous_region_index, optimal_edges, erroneous_edges, common_edges

    # If no points have been added so far, none can be removed
    if data_points is None:
        plt.title("Place at least 2 points.")
        is_mst_computed = False

        # Update the canvas
        canvas.draw()
        return

    # If there is more than one point
    if data_points.shape[0] > 1:

        # Remove the last point (row) from data_points
        data_points = data_points[:-1, :]

        # Delete the current tree from interim MSTs
        interim_msts = interim_msts[:-1]

        # Clear the whole plot
        ax.clear()

        # Reset the final mst and its edges to the last
        final_mst = interim_msts[-1][1]
        final_mst_edges = interim_msts[-1][0]

        # Reset any existing subdivision, transitions, wrong regions, and edges from wrong trees
        sub_division = None

        correct_transitions = []
        correct_transition_index = -1

        erroneous_regions = []
        erroneous_region_index = -1

        erroneous_edges = None
        optimal_edges = None
        common_edges = None

        # Plot the nodes and edges of the last MST
        ax.plot([data_points[i][0] for i in range(data_points.shape[0])],
                [data_points[i][1] for i in range(data_points.shape[0])],
                "ok", markersize=5)

        line_segments = mc.LineCollection(final_mst_edges, colors="black", linewidths=2)

        ax.add_collection(line_segments)

        # Disable autoscale to prevent the coordinate from involuntary moving
        plt.autoscale(enable=False, axis='both', tight=None)

        ax.margins(0.1)

        # Set the title to the old weight
        ax.set_title("Tree Weight: " + str(np.round(interim_msts[-1][2], 2)) + " cm")

        # Reframe the coordinate system to fit the old MST
        center_mst()

        # Disable autoscale to prevent the coordinate from involuntary moving
        plt.autoscale(enable=False, axis='both', tight=None)

    # If there is only one data point, clear everything
    else:
        clear_all()

    # Update the canvas
    canvas.draw()


def undo_line():
    """
    Removes extended, starting- and end point of the drawn line.
    :return: None
    """
    global is_extended, starting_point, starting_coordinate, end_point, end_coordinate, line, \
        extended_starting_point, extended_starting_coordinate, extended_end_point, extended_end_coordinate, extended_line, \
        additional_point, mst_plot, final_mst_edges, sub_division, correct_transitions, correct_transition_index, \
        is_subdivision_computed, shows_correct_regions, compute_regions_button, draw_line_button, cid

    # Reset the transitions and the index
    correct_transitions = []
    correct_transition_index = -1

    # No line can be removed for None or one data point
    # Sanity check
    if data_points is None:
        plt.title("Place at least 2 points.")
        is_mst_computed = False

        # Update the canvas
        canvas.draw()
        return

    if len(data_points) < 2:
        ax.set_title("Place at least 2 points.")

        # Update the canvas
        canvas.draw()
        return

    # Remove an additional point, if once been added to the plot
    if additional_point is not None:
        additional_point = additional_point.pop(0)
        additional_point.remove()
        additional_point = None

    # Remove the current MST plot
    mst_plot.remove()

    line_segments = mc.LineCollection(final_mst_edges, colors="black", linewidths=2)

    mst_plot = ax.add_collection(line_segments)

    # When extended, return to the 'unexpanded' line
    if is_extended:
        is_extended = False

        # Remove the extended line
        extended_line = extended_line.pop(0)
        extended_line.remove()

        # Remove the extended starting- and end point
        extended_starting_point = extended_starting_point.pop(0)
        extended_starting_point.remove()
        extended_starting_point, extended_starting_coordinate = None, None

        extended_end_point = extended_end_point.pop(0)
        extended_end_point.remove()
        extended_end_point, extended_end_coordinate = None, None

        # Plot again the previous starting- and end point
        starting_point = ax.plot(starting_coordinate.x, starting_coordinate.y, 'ko', mfc='none', markersize=5)
        end_point = ax.plot(end_coordinate.x, end_coordinate.y, 'ko', mfc='none', markersize=5)

        # And the line connecting both
        x_coordinates = [starting_coordinate.x, end_coordinate.x]
        y_coordinates = [starting_coordinate.y, end_coordinate.y]
        line = ax.plot(x_coordinates, y_coordinates, 'ko', linestyle="--", mfc='none', markersize=5)

        # Re-adjust the coordinate system to the 'shrinked' line
        center_mst()

        # Rename the draw_line_button and enable it
        draw_line_button.configure(text="Extend Line", command=extend_line, state="active")

        # Call plot next such that the MST on display includes the starting point
        plot_next_correct_transition()

        # Disable autoscale to prevent the coordinate from involuntary moving
        plt.autoscale(enable=False, axis='both', tight=None)

    # If there is a starting point but no end point, remove the starting point
    elif starting_point is not None and end_point is None:

        starting_point = starting_point.pop(0)
        starting_point.remove()
        starting_point, starting_coordinate = None, None

        # Change instructions in plot title
        ax.set_title("Place a starting point.")

        # Adjust coordinate system accordingly
        center_mst()

        # Disable autoscale to prevent the coordinate from involuntary moving
        plt.autoscale(enable=False, axis='both', tight=None)

    # If both starting- and end point are present, remove the line and the end point
    elif starting_point is not None and end_point is not None:

        line = line.pop(0)
        line.remove()

        end_point = end_point.pop(0)
        end_point.remove()
        end_point, end_coordinate = None, None

        # Rename draw_line_button (before: "Extend Line")
        draw_line_button.configure(text="Draw Line", state="disabled")

        # Disable both plot_* buttons
        plot_next_button.configure(state="disabled")
        plot_prev_button.configure(state="disabled")

        # Adjust instructions
        ax.set_title("Place an end point.")

        # Reframe coordinate system
        center_mst()

        # Disable autoscale to prevent the coordinate from involuntary moving
        plt.autoscale(enable=False, axis='both', tight=None)

    # If both points are None, go back to building the MST
    elif starting_point is None and end_point is None:

        # Remove everything from the plot
        ax.clear()

        # Set the undo_button such that the last node from the MST is removed
        undo_button.configure(command=undo_last_node)

        # Plot again data points and edges
        ax.plot([data_points[i][0] for i in range(data_points.shape[0])],
                [data_points[i][1] for i in range(data_points.shape[0])],
                "ok", markersize=5)
        line_segments = mc.LineCollection(final_mst_edges, colors="black", linewidths=2)

        mst_plot = ax.add_collection(line_segments)

        # Reset subdivision and its dependent booleans
        sub_division = None
        shows_correct_regions = False
        is_subdivision_computed = False

        # Rename compute_regions_button
        compute_regions_button.configure(text="Show MST Regions")

        # Rename and reset the draw_line_button and the plot_* buttons
        draw_line_button.configure(text="Draw Line", command=ready_to_draw, state="disabled")
        plot_next_button.configure(state="disabled")
        plot_prev_button.configure(state="disabled")

        ax.set_title("Place points by mouse click.")

        # Make canvas responsive
        cid = fig.canvas.mpl_connect('button_press_event', onclick)

        # Reframe the coordinate system
        center_mst()

    # Update the canvas
    canvas.draw()


def undo_errors():
    """
    Removes the plots for an erronious region.
    :return: None
    """
    global mst_plot, additional_point, erroneous_edges, optimal_edges, common_edges, sub_division, \
        shows_correct_regions, is_subdivision_computed, compute_regions_button, draw_line_button, undo_button, \
        plot_next_button, plot_prev_button, cid

    # No wrong region can be removed for None or one data point
    # Sanity check
    if data_points is None:
        plt.title("Place at least 2 points.")
        is_mst_computed = False

        # Update the canvas
        canvas.draw()
        return

    if len(data_points) < 2:
        ax.set_title("Place at least 2 points.")

        # Update the canvas
        canvas.draw()
        return

    # If all edges are set to None
    # i.e. the undo-Button has been clicked twice
    # Regions are removed and points can be added again
    if (erroneous_edges is None and optimal_edges is None and common_edges is None) or \
            sub_division.area_of_erroneous_regions == 0.0:

        # Remove all plots
        ax.clear()

        # Redirect the undo_button such that the last node is removed
        undo_button.configure(command=undo_last_node)

        # Plot again the data points and the edges of the MST
        ax.plot([data_points[i][0] for i in range(data_points.shape[0])],
                [data_points[i][1] for i in range(data_points.shape[0])],
                "ok", markersize=5)

        line_segments = mc.LineCollection(final_mst_edges, colors="black", linewidths=2)
        mst_plot = ax.add_collection(line_segments)

        # Reset subdivision and dependent boolean variables
        sub_division = None
        shows_correct_regions = False
        is_subdivision_computed = False

        # Rename buttons and set their state accordingly
        compute_regions_button.configure(text="Show MST Regions")

        draw_line_button.configure(text="Draw Line", command=ready_to_draw, state="disabled")
        plot_prev_button.configure(state="disabled")
        plot_next_button.configure(state="disabled")

        # Reframe the coordinate system
        center_mst()

        # Make canvas responsive
        cid = fig.canvas.mpl_connect('button_press_event', onclick)

        ax.set_title("Place points by mouse click.")

        # Update the canvas
        canvas.draw()
        return
    else:
        # Remove the additional point from the last region, if there is one
        if additional_point is not None:
            additional_point = additional_point.pop(0)
            additional_point.remove()
            additional_point = None

        # Remove any remaining erroneous edges from the last region
        if erroneous_edges is not None:
            erroneous_edges.remove()
            erroneous_edges = None

        # Remove any remaining optimal edges from the last region
        if optimal_edges is not None:
            optimal_edges.remove()
            optimal_edges = None

        # Remove any remaining common edges from the last region
        if common_edges is not None:
            common_edges.remove()
            common_edges = None

        # Plot again the data points and the edges of the MST
        ax.plot([data_points[i][0] for i in range(data_points.shape[0])],
                [data_points[i][1] for i in range(data_points.shape[0])],
                "ok", markersize=5)

        line_segments = mc.LineCollection(final_mst_edges, colors="black", linewidths=2)
        mst_plot = ax.add_collection(line_segments)

        ax.set_title("Area of false Regions: " + str(np.round(sub_division.area_of_erroneous_regions, 2)) + r" cm$^2$" + "\n" +
                     "Total Area: " + str(np.round(sub_division.total_area, 2)) + r" cm$^2$")

        # Reframe coordinate system around the MST
        center_mst()

        # Disconnect canvas
        canvas.mpl_disconnect(cid)

        # Update the canvas
        canvas.draw()


def clear_all():
    """
    Clears all plots and resets all variables.
    """
    global data_points, sub_division, correct_transitions, correct_transition_index, \
        erroneous_regions, erroneous_region_index, erroneous_edges, optimal_edges, common_edges, \
        interim_msts, final_mst, mst_plot, \
        additional_point, final_mst_edges, keys_to_edge_index, zoom_position,\
        zoom_ratio, is_subdivision_computed, is_extended, cid, \
        starting_coordinate, end_coordinate, starting_point, end_point, line,\
        extended_starting_coordinate, extended_end_coordinate, extended_starting_point, extended_end_point, extended_line,\
        draw_line_button, plot_next_button, plot_prev_button, shows_correct_regions

    # Clear all plots and reset coordinate system
    ax.clear()

    # Equal Ratio of x- and y-axis
    ax.set_aspect('equal', 'box')

    # Reset all variables to their initial values
    original_xmin, original_xmax = -10, 10
    ax.set_xlim([original_xmin, original_xmax])

    original_ymin, original_ymax = -10, 10
    ax.set_ylim([original_ymin, original_ymax])

    ax.set_title("Place points by mouse click.")

    # Disable autoscale to prevent the coordinate from involuntary moving
    plt.autoscale(enable=False, axis='both', tight=None)

    # Last zoom position
    zoom_position = (original_xmin, original_xmax, original_ymin, original_ymax)

    # Data structures for the MST
    data_points = None
    interim_msts = []
    final_mst = None
    mst_plot = None
    additional_point = None

    # Additional edge types needed to plot the wrong and optimal MST at the same time
    common_edges = None
    optimal_edges = None
    erroneous_edges = None

    # Allows faster plotting of correct_transitions
    final_mst_edges = []
    keys_to_edge_index = []

    sub_division = None
    correct_transitions = []
    correct_transition_index = -1

    erroneous_regions = []
    erroneous_region_index = -1

    # Line variables
    starting_coordinate, end_coordinate = None, None
    starting_point, end_point = None, None
    line = None

    extended_starting_coordinate, extended_end_coordinate = None, None
    extended_starting_point, extended_end_point = None, None
    extended_line = None

    # Boolean variables to keep track where the User is
    is_subdivision_computed = False

    # Boolean to indicate where the user currently is
    shows_correct_regions = False

    is_line_drawn = False
    is_extended = False

    is_paused = False

    # Pointer to the Help-Window
    help_window = None

    # Reset buttons
    undo_button.configure(command=undo_last_node)
    compute_regions_button.configure(text="Show MST Regions", command=show_regions_handler)
    draw_line_button.configure(text="Draw Line", command=ready_to_draw, state="disabled")
    plot_next_button.configure(command=plot_next_handler, state="disabled")
    plot_prev_button.configure(command=plot_prev_handler, state="disabled")

    # Make canvas responsive
    cid = fig.canvas.mpl_connect('button_press_event', onclick)

    # Disable autoscale to prevent the coordinate from involuntary moving
    plt.autoscale(enable=False, axis='both', tight=None)

    # Update the canvas
    canvas.draw()


def _quit():
    """
    Quits the application.
    """

    # Stop the mainloop
    root.quit()

    # Destroy the window
    # This is necessary on Windows to prevent fatal python Error: PyEval_RestoreThread: NULL tstate
    root.destroy()


def pause():
    """
    Pauses the responsiveness of the canvas.
    Makes matplotlib's own navigation tools useable, such as 'Zoom to Rectangle' or 'Pan Axes'.
    :return:
    """
    global is_paused, cid

    # If currently not paused, disconnect canvas
    if not is_paused:
        canvas.mpl_disconnect(cid)
        pause_button.configure(text="Resume")
        is_paused = True

    # Otherwise, connect canvas again
    else:
        cid = fig.canvas.mpl_connect('button_press_event', onclick)
        pause_button.configure(text="Pause")
        is_paused = False

    # Update the canvas
    canvas.draw()


def open_help_window():
    """
    Opens/ Closes a help window.
    :return:
    """
    global help_button, is_help_open, help_window

    # If help window is open, destroy it
    if help_window is not None:
        help_button.configure(text="Help")

        help_window.destroy()
        help_window = None
    # Otherwise, open a window with all necessary information about the functions of the application
    else:
        help_button.configure(text="Close Help")

        help_window = Toplevel(root)
        help_window.wm_title("Help")

        zoom_out_bt = Label(help_window, text= "Zoom out\t", font='Helvetica 10 bold')
        zoom_out_bt.grid(row=0, column=0, sticky="w", pady=(7, 15), padx=20)
        zoom_out_exp = Label(help_window, text="\tZooms out by factor 1.15.")
        zoom_out_exp.grid(row=0, column=1, sticky="w", pady=(7, 15), padx=20)

        center_diag_bt = Label(help_window, text="Center Diagram\t", font='Helvetica 10 bold')
        center_diag_bt.grid(row=1, column=0, sticky="w", pady=15, padx=20)
        center_diag_bt = Label(help_window, text="\tCenters on the diagram.")
        center_diag_bt.grid(row=1, column=1, sticky="w", pady=15, padx=20)

        center_mst_bt = Label(help_window, text="Center MST\t", font='Helvetica 10 bold')
        center_mst_bt.grid(row=2, column=0, sticky="w", pady=15, padx=20)
        center_mst_exp = Label(help_window, text="\tCenters on the MST.")
        center_mst_exp.grid(row=2, column=1, sticky="w", pady=15, padx=20)

        clear_all_bt = Label(help_window, text="Clear All\t", font='Helvetica 10 bold')
        clear_all_bt.grid(row=3, column=0, sticky="w", pady=15, padx=20)
        clear_all_exp = Label(help_window, text="\tClears the coordinate system completely.")
        clear_all_exp.grid(row=3, column=1, sticky="w", pady=15, padx=20)

        undo_bt = Label(help_window, text="Undo\t", font='Helvetica 10 bold')
        undo_bt.grid(row=4, column=0, sticky="w", pady=15, padx=20)
        undo_exp = Label(help_window, text="\tUndoes the last Action.")
        undo_exp.grid(row=4, column=1, sticky="w", pady=15, padx=20)

        show_region_bt = Label(help_window, text="Show MST Regions\t", font='Helvetica 10 bold')
        show_region_bt.grid(row=5, column=0, sticky="w", pady=15, padx=20)
        show_region_exp = Label(help_window, text="\tDisplays regions in which the MST has the same topology.")
        show_region_exp.grid(row=5, column=1, sticky="w", pady=15, padx=20)

        show_region_bt2 = Label(help_window, text="Show False Regions\t", font='Helvetica 10 bold')
        show_region_bt2.grid(row=6, column=0, sticky="w", pady=15, padx=20)
        show_region_exp2 = Label(help_window, text="\tDisplays regions in which the algorithm produces a false topology.")
        show_region_exp2.grid(row=6, column=1, sticky="w", pady=15, padx=20)

        pause_bt = Label(help_window, text="Pause\t", font='Helvetica 10 bold')
        pause_bt.grid(row=7, column=0, sticky="w", padx=20, pady=(15, 0))
        pause_exp1 = Label(help_window, text="\tPauses the Responsiveness of the Coordinate System.")
        pause_exp1.grid(row=7, column=1, sticky="w", padx=20, pady=(15, 0))
        # pause_exp2 = Label(help_window, text="\tWhen clicked, matplotlibs' own tool bar can be used,")
        pause_exp2 = Label(help_window, text="\tWhen clicked, 'Pan Axes' or 'Zoom to Rectangle' can be used.")
        pause_exp2.grid(row=8, column=1, sticky="w", padx=20, pady=(0, 15))
        # pause_exp3 = Label(help_window, text="\te.g., 'Pan Axes' or 'Zoom to Rectangle'.")
        # pause_exp3.grid(row=9, column=1, sticky="w", padx=20, pady=(0, 15))

        resume_bt = Label(help_window, text="Resume\t", font='Helvetica 10 bold')
        resume_bt.grid(row=9, column=0, sticky="w", pady=15, padx=20)
        resume_exp1 = Label(help_window, text="\tRe-Activates the Responsiveness of the Coordinate System.")
        resume_exp1.grid(row=9, column=1, sticky="w", pady=15, padx=20)

        quit_bt = Label(help_window, text="Quit\t", font='Helvetica 10 bold')
        quit_bt.grid(row=10, column=0, sticky="w", pady=15, padx=20)
        quit_exp = Label(help_window, text="\tCloses the application.")
        quit_exp.grid(row=10, column=1, sticky="w", pady=15, padx=20)

        plot_prev_bt = Label(help_window, text="Plot Prev\t", font='Helvetica 10 bold')
        plot_prev_bt.grid(row=11, column=0, sticky="w", pady=15, padx=20)
        plot_prev_exp = Label(help_window, text="\tPlots the previous MST transition/ false region.")
        plot_prev_exp.grid(row=11, column=1, sticky="w", pady=15, padx=20)

        draw_line_bt = Label(help_window, text="Draw Line\t", font='Helvetica 10 bold')
        draw_line_bt.grid(row=12, column=0, sticky="w", pady=15, padx=20)
        draw_line_exp = Label(help_window, text="\tPlaces a Line between to selected points.")
        draw_line_exp.grid(row=12, column=1, sticky="w", pady=15, padx=20)

        extend_line_bt = Label(help_window, text="Extend Line\t", font='Helvetica 10 bold')
        extend_line_bt.grid(row=13, column=0, sticky="w", pady=15, padx=20)
        extend_line_exp = Label(help_window, text="\tExpands the line to the boundaries of the diagram.")
        extend_line_exp.grid(row=13, column=1, sticky="w", pady=15, padx=20)

        plot_next_bt = Label(help_window, text="Plot Next\t", font='Helvetica 10 bold')
        plot_next_bt.grid(row=14, column=0, sticky="w", pady=15, padx=20)
        plot_next_exp = Label(help_window, text="\tPlots the next MST transition/ false region.")
        plot_next_exp.grid(row=14, column=1, sticky="w", pady=15, padx=20)

        attention = Label(help_window, text="In rare cases, rounding errors can lead to falsely computed topologies and regions, or cause the GUI to fail.",
                          font='Helvetica 10 bold')
        attention.grid(row=15, column=0, columnspan=2, sticky="ew", pady=(3, 5), padx=20)


# First Row of Buttons
zoom_out_button = tkinter.Button(master=root, text="Zoom Out", command=zoom_out, width = 25, bd=3)
zoom_out_button.grid(row=2, columnspan=2, padx=5, pady=15, column=0 , sticky="ew")

center_diagram_button = tkinter.Button(master=root, text="Center Diagram", command=center_diagram, bd=3)
center_diagram_button.grid(row=2, padx=5, pady=15, column= 2, sticky="ew")

center_mst_button = tkinter.Button(master=root, text="Center MST", command=center_mst, width = 25, bd=3)
center_mst_button.grid(row=2, columnspan=2, padx=5, pady=15, column=3, sticky="ew")

# Second Row
clear_button = tkinter.Button(master=root, text="Clear All", command=clear_all, width = 10, bd=3)
clear_button.grid(row=3, padx=5, pady=0, column=0, sticky="ew")

undo_button = tkinter.Button(master=root, text="Undo", command=undo_last_node, width = 10, bd=3)
undo_button.grid(row=3, padx=5, pady=0, column=1, sticky="ew")

compute_regions_button = tkinter.Button(master=root, text="Show MST Regions", command=show_regions_handler, bd=3)
compute_regions_button.grid(row=3, padx=5, pady=0, column=2, sticky="ew")

pause_button = tkinter.Button(master=root, text="Pause", command=pause, width=10, bd=3)
pause_button.grid(row=3, padx=5, pady=0, column=3, sticky="ew")

quit_button = tkinter.Button(master=root, text="Quit", command=_quit, width=10, bd=3)
quit_button.grid(row=3, padx=5, pady=0, column=4, sticky="ew")

# Third Row
plot_prev_button = tkinter.Button(master=root, text="Plot Prev", command=plot_prev_handler, state="disabled",
                                  width = 25, bd=3)
plot_prev_button.grid(row=4, columnspan=2, padx=5, pady=15, column=0, sticky="ew")

draw_line_button = tkinter.Button(master=root, text="Draw Line", command=ready_to_draw, state="disabled", bd=3)
draw_line_button.grid(row=4, padx=5, pady=15, column=2, sticky="ew")

plot_next_button = tkinter.Button(master=root, text="Plot Next", command=plot_next_handler, state="disabled",
                                  width = 25, bd=3)
plot_next_button.grid(row=4, columnspan=2, padx=5, pady=15, column=3, sticky="ew")

# Help Button in the last line
help_button = tkinter.Button(master=root, text="Help", command=open_help_window, bd=3)
help_button.grid(row=5, column=0, padx=5, pady=(0, 10), columnspan=5, sticky="ew")

canvas.mpl_connect("key_press_event", on_key_press)
cid = fig.canvas.mpl_connect('button_press_event', onclick)

tkinter.mainloop()

# If you put root.destroy() here, it will cause an error if the window is
# closed with the window manager.
