import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

class ContainerLoadingOptimizer:
    """Container Loading Optimization for boards with various orientations
    This function contains information about the boards, their dimensions, and the container's dimensions.
    It also has a packing algorithm, residual space management, and statistical calculations.
    It also provides methods to visualize the loading process and compare different starting orientations.
    """

    def __init__(self, container_dims, boards):
        self.container = container_dims
        self.boards = boards
        self.orientations = [
            (0, 1, 2), (1, 0, 2), (0, 2, 1),
            (2, 0, 1), (1, 2, 0), (2, 1, 0)
        ]
        self.orientation_names = [
            "Flat on floor - long edge along the container length",
            "Flat on floor - short edge along the container length",
            "Vertical - width upright, long edge along the container length",
            "Vertical - width upright, long edge along the container width",
            "Vertical - length upright, short edge along the container length",
            "Vertical - length upright, short edge along the container width"
        ]
        self.loaded_boards = []
        self.residual_spaces = []

    def get_oriented_dims(self, board_dims, orientation):
        return tuple(board_dims[i] for i in orientation)

    def calculate_stacks_in_space(self, space_dims, board_dims_oriented, orientation_index):
        """
        Calculate how many stacks can fit in a given space, with robust gap logic for flat orientations.
        """
        if any(space_dims[i] < board_dims_oriented[i] for i in range(3)):
            return 0, 0, 0, 0

        stacks_x = space_dims[0] // board_dims_oriented[0] if board_dims_oriented[0] > 0 else 0
        stacks_y = space_dims[1] // board_dims_oriented[1] if board_dims_oriented[1] > 0 else 0
        
        board_height = board_dims_oriented[2]
        if board_height <= 0:
            return int(stacks_x), int(stacks_y), 0, 0

        if orientation_index in [0, 1]:
            available_height = space_dims[2]
            side_height_limit = self.container.get('side_height', self.container['H'])
            max_boards_in_set = self.container.get('max_boards_stack', 20)
            gap_height = board_height * 7

            max_boards_simple_est = min(
                available_height // board_height if board_height > 0 else 0,
                side_height_limit // board_height if board_height > 0 else float('inf')
            )

            for num_boards in range(int(max_boards_simple_est), 0, -1):
                num_gaps = (num_boards - 1) // max_boards_in_set if num_boards > 1 else 0
                
                total_stack_height = (num_boards * board_height) + (num_gaps * gap_height)

                if total_stack_height <= available_height and total_stack_height <= side_height_limit:
                     boards_per_stack = num_boards
                     used_z = total_stack_height
                     return int(stacks_x), int(stacks_y), int(boards_per_stack), used_z

            return int(stacks_x), int(stacks_y), 0, 0

        else:
            available_height = space_dims[2]
            side_height_limit = self.container.get('side_height', self.container['H'])
            
            max_boards_geom = available_height // board_height if board_height > 0 else 0
            
            if board_height > side_height_limit:
                boards_per_stack = 1 if max_boards_geom >= 1 else 0
            else:
                max_boards_by_rule = side_height_limit // board_height if board_height > 0 else float('inf')
                boards_per_stack = min(max_boards_geom, int(max_boards_by_rule))
            
            used_z = boards_per_stack * board_height
            return int(stacks_x), int(stacks_y), int(boards_per_stack), used_z

    def find_best_orientation_for_space(self, space_dims, board_type):
        """Find the best orientation for a board type in a given space"""
        best_count = 0
        best_config = None
        for i, orientation in enumerate(self.orientations):
            oriented_dims = self.get_oriented_dims(board_type["dims"], orientation)
            stacks_x, stacks_y, boards_per_stack, used_z = self.calculate_stacks_in_space(space_dims, oriented_dims, i)
            total_boards = stacks_x * stacks_y * boards_per_stack
            
            if total_boards > best_count:
                best_count = total_boards
                best_config = {
                    "orientation": orientation, "orientation_name": self.orientation_names[i],
                    "oriented_dims": oriented_dims, "stacks_x": stacks_x, "stacks_y": stacks_y,
                    "boards_per_stack": boards_per_stack, "total_boards": total_boards,
                    "orientation_index": i, "used_z": used_z
                }
        return best_config

    def find_specific_orientation_for_space(self, space_dims, board_type, forced_orientation_idx):
        """Find configuration for a specific orientation in a given space"""
        orientation = self.orientations[forced_orientation_idx]
        oriented_dims = self.get_oriented_dims(board_type["dims"], orientation)
        stacks_x, stacks_y, boards_per_stack, used_z = self.calculate_stacks_in_space(space_dims, oriented_dims, forced_orientation_idx)
        total_boards = stacks_x * stacks_y * boards_per_stack
        
        if total_boards > 0:
            return {
                "orientation": orientation, "orientation_name": self.orientation_names[forced_orientation_idx],
                "oriented_dims": oriented_dims, "stacks_x": stacks_x, "stacks_y": stacks_y,
                "boards_per_stack": boards_per_stack, "total_boards": total_boards,
                "orientation_index": forced_orientation_idx, "used_z": used_z
            }
        return None

    def create_residual_spaces(self, space_dims, placed_config, space_position):
        """Create residual spaces after placing boards, using the correct total height."""
        residual_spaces = []
        oriented_dims = placed_config["oriented_dims"]
        stacks_x, stacks_y = placed_config["stacks_x"], placed_config["stacks_y"]
        used_x = stacks_x * oriented_dims[0]
        used_y = stacks_y * oriented_dims[1]
        used_z = placed_config["used_z"] 

        if space_dims[0] > used_x:
            residual_spaces.append({
                "dims": (space_dims[0] - used_x, space_dims[1], space_dims[2]),
                "position": (space_position[0] + used_x, space_position[1], space_position[2])
            })
        if space_dims[1] > used_y:
            residual_spaces.append({
                "dims": (used_x, space_dims[1] - used_y, space_dims[2]),
                "position": (space_position[0], space_position[1] + used_y, space_position[2])
            })
        # if space_dims[2] > used_z:
        #     residual_spaces.append({
        #         "dims": (used_x, used_y, space_dims[2] - used_z),
        #         "position": (space_position[0], space_position[1], space_position[2] + used_z)
        #     })
        return residual_spaces

    def optimize_loading_with_forced_first_orientation(self, forced_first_orientation=None):
        """Main optimization algorithm with optional forced first orientation"""
        self.loaded_boards = []
        self.residual_spaces = []
        clearance_height = self.container.get('ground_clearance', 0)
        
        if clearance_height >= self.container["H"]:
            print("Warning: Ground clearance is greater than or equal to container height. No loading is possible.")
            return 0, []

        initial_dims = (self.container["L"], self.container["W"], self.container["H"] - clearance_height)
        initial_position = (0, 0, clearance_height)
        available_spaces = [{"dims": initial_dims, "position": initial_position}]
        # available_spaces = [{"dims": (self.container["L"], self.container["W"], self.container["H"]), "position": (0, 0, 0)}]
        total_loaded = 0
        remaining_boards = {board["id"]: board["quantity"] for board in self.boards}
        total_weight_loaded = 0
        max_weight = self.container.get("max_weight", float('inf'))
        first_placement = True

        while available_spaces and any(remaining_boards.values()):
            available_spaces.sort(key=lambda x: x["dims"][0] * x["dims"][1] * x["dims"][2], reverse=True)
            space = available_spaces.pop(0)
            best_placement = None
            best_board_type = None

            for board_type in self.boards:
                if remaining_boards[board_type["id"]] <= 0:
                    continue

                config = self.find_specific_orientation_for_space(space["dims"], board_type, forced_first_orientation) if first_placement and forced_first_orientation is not None else self.find_best_orientation_for_space(space["dims"], board_type)

                if config and config["total_boards"] > 0:
                    possible_boards = min(config["total_boards"], remaining_boards[board_type["id"]])
                    board_weight = board_type.get("weight", 0)
                    if board_weight > 0:
                        remaining_weight_capacity = max_weight - total_weight_loaded
                        max_boards_by_weight = int(remaining_weight_capacity // board_weight) if board_weight > 0 else float('inf')
                        actual_boards = min(possible_boards, max_boards_by_weight)
                    else:
                        actual_boards = possible_boards

                    if actual_boards <= 0:
                        continue
                    
                    config["actual_boards_for_eval"] = actual_boards
                    if not best_placement or config["actual_boards_for_eval"] > best_placement.get("actual_boards_for_eval", 0):
                        best_placement = config
                        best_board_type = board_type

            if best_placement:
                actual_boards_to_load = best_placement.pop("actual_boards_for_eval")
                best_placement["actual_boards"] = actual_boards_to_load
                
                self.loaded_boards.append({
                    "board_type": best_board_type,
                    "config": best_placement,
                    "position": space["position"]
                })
                
                remaining_boards[best_board_type["id"]] -= actual_boards_to_load
                total_loaded += actual_boards_to_load
                total_weight_loaded += actual_boards_to_load * best_board_type.get("weight", 0)
                
                residual_spaces = self.create_residual_spaces(space["dims"], best_placement, space["position"])
                available_spaces.extend(residual_spaces)
                first_placement = False
            else:
                break
        return total_loaded, self.loaded_boards


    def optimize_loading(self):
        """Original optimization algorithm (for backward compatibility)"""
        return self.optimize_loading_with_forced_first_orientation(None)

    def calculate_statistics(self):
        """Calculate detailed loading statistics including weight"""
        total_boards = sum(placement["config"]["actual_boards"] for placement in self.loaded_boards)
        
        usable_height = self.container.get('side_height', self.container['H'])
        container_volume_usable = self.container["L"] * self.container["W"] * usable_height
        ground_clearance = self.container.get('ground_clearance', 0)
        container_volume = self.container["L"] * self.container["W"] * self.container["H"]

        board_volume_clipped = 0
        gap_volume_clipped = 0

        for placement in self.loaded_boards:
            blocks = self.get_block_structure(placement)
            
            xy_stacks = {}
            for block in blocks:
                key = (block['position'][0], block['position'][1])
                if key not in xy_stacks:
                    xy_stacks[key] = []
                xy_stacks[key].append(block)

            for xy_pos, stack_blocks in xy_stacks.items():
                stack_blocks.sort(key=lambda b: b['position'][2])

                for i, block in enumerate(stack_blocks):
                    block_pos, block_dims = block["position"], block["dimensions"]
                    z_bottom, z_top = block_pos[2], block_pos[2] + block_dims[2]

                    clipped_board_height = max(0, min(usable_height, z_top) - z_bottom)
                    board_volume_clipped += block_dims[0] * block_dims[1] * clipped_board_height
                    
                    if i < len(stack_blocks) - 1:
                        next_block = stack_blocks[i+1]
                        gap_z_bottom = z_top
                        gap_z_top = next_block['position'][2]
                        
                        if gap_z_top > gap_z_bottom: # A real gap exists
                            gap_width, gap_depth = block_dims[0], block_dims[1]
                            clipped_gap_height = max(0, min(usable_height, gap_z_top) - gap_z_bottom)
                            gap_volume_clipped += gap_width * gap_depth * clipped_gap_height
        
        ground_clearance_volume_clipped = self.container["L"] * self.container["W"] * min(ground_clearance, usable_height)

        board_volume = sum(
            placement["config"]["actual_boards"] *
            (placement["board_type"]["dims"][0] * placement["board_type"]["dims"][1] * placement["board_type"]["dims"][2])
            for placement in self.loaded_boards
        )
        total_utilized_volume = board_volume_clipped + gap_volume_clipped + ground_clearance_volume_clipped
        volume_utilization = (total_utilized_volume / container_volume_usable) * 100 if container_volume_usable > 0 else 0
        total_weight = sum(
            placement["config"]["actual_boards"] * placement["board_type"].get("weight", 0)
            for placement in self.loaded_boards
        )
        max_weight = self.container.get("max_weight", float('inf'))
        weight_utilization = (total_weight / max_weight) * 100 if max_weight != float('inf') else 0
        orientation_stats = {}
        for placement in self.loaded_boards:
            orientation_name = placement["config"]["orientation_name"]
            if orientation_name not in orientation_stats:
                orientation_stats[orientation_name] = {"boards": 0, "stacks": 0, "volume": 0, "weight": 0}
            orientation_stats[orientation_name]["boards"] += placement["config"]["actual_boards"]
            orientation_stats[orientation_name]["stacks"] += placement["config"]["stacks_x"] * placement["config"]["stacks_y"]
            orientation_stats[orientation_name]["volume"] += (placement["config"]["actual_boards"] * (placement["board_type"]["dims"][0] * placement["board_type"]["dims"][1] * placement["board_type"]["dims"][2]))
            orientation_stats[orientation_name]["weight"] += (placement["config"]["actual_boards"] * placement["board_type"].get("weight", 0))
        board_type_stats = {}
        for placement in self.loaded_boards:
            board_id = placement["board_type"]["id"]
            if board_id not in board_type_stats:
                board_type_stats[board_id] = {"loaded": 0, "total": placement["board_type"]["quantity"], "utilization": 0, "weight": 0}
            board_type_stats[board_id]["loaded"] += placement["config"]["actual_boards"]
            board_type_stats[board_id]["weight"] += (placement["config"]["actual_boards"] * placement["board_type"].get("weight", 0))
        for board_id in board_type_stats:
            if board_type_stats[board_id]["total"] > 0:
                board_type_stats[board_id]["utilization"] = (board_type_stats[board_id]["loaded"] / board_type_stats[board_id]["total"]) * 100
        return {
            "total_boards": total_boards, "container_volume": container_volume, "board_volume": board_volume,
            "volume_utilization": volume_utilization, "total_weight": total_weight, "max_weight": max_weight,
            "weight_utilization": weight_utilization, "orientation_stats": orientation_stats, "board_type_stats": board_type_stats
        }

    def create_block_mesh(self, x_pos, y_pos, z_pos, width, depth, height):
        """Create a mesh for a block of boards"""
        vertices = np.array([
            [x_pos, y_pos, z_pos], [x_pos + width, y_pos, z_pos], [x_pos + width, y_pos + depth, z_pos],
            [x_pos, y_pos + depth, z_pos], [x_pos, y_pos, z_pos + height], [x_pos + width, y_pos, z_pos + height],
            [x_pos + width, y_pos + depth, z_pos + height], [x_pos, y_pos + depth, z_pos + height]
        ])
        faces = [
            [0, 1, 2], [0, 2, 3], [4, 7, 6], [4, 6, 5], [0, 4, 5], [0, 5, 1],
            [2, 6, 7], [2, 7, 3], [0, 3, 7], [0, 7, 4], [1, 5, 6], [1, 6, 2]
        ]
        return vertices, faces

    def get_block_structure(self, placement):
        """Determine how to structure blocks based on orientation, creating visual gaps for flat stacks."""
        config = placement["config"]
        orientation_idx = config["orientation_index"]
        oriented_dims = config["oriented_dims"]
        position = placement["position"]
        blocks = []

        if orientation_idx in [0, 1]:
            max_in_set = self.container.get('max_boards_stack', 20)
            thickness = oriented_dims[2]
            gap_height = thickness * 7
            
            for bx in range(config["stacks_x"]):
                for by in range(config["stacks_y"]):
                    start_board_index_in_placement = (bx * config["stacks_y"] + by) * config["boards_per_stack"]
                    if start_board_index_in_placement >= config["actual_boards"]:
                        continue
                    
                    boards_in_this_xy_stack = min(config["boards_per_stack"], config["actual_boards"] - start_board_index_in_placement)
                    
                    x_pos = position[0] + bx * oriented_dims[0]
                    y_pos = position[1] + by * oriented_dims[1]
                    current_z_pos = position[2]
                    boards_processed = 0
                    
                    while boards_processed < boards_in_this_xy_stack:
                        boards_in_this_set = min(max_in_set, boards_in_this_xy_stack - boards_processed)
                        block_height = boards_in_this_set * thickness
                        
                        blocks.append({
                            "position": (x_pos, y_pos, current_z_pos),
                            "dimensions": (oriented_dims[0], oriented_dims[1], block_height),
                            "boards": boards_in_this_set, "type": "length_width_block"
                        })
                        
                        boards_processed += boards_in_this_set
                        current_z_pos += block_height
                        if boards_processed < boards_in_this_xy_stack: 
                             current_z_pos += gap_height

        elif orientation_idx in [2, 4]:
            for bx in range(config["stacks_x"]):
                for bz in range(config["boards_per_stack"]):
                    x_pos, y_pos, z_pos = position[0] + bx * oriented_dims[0], position[1], position[2] + bz * oriented_dims[2]
                    start_board_index = (bx * config["boards_per_stack"] + bz) * config["stacks_y"]
                    if start_board_index < config["actual_boards"]:
                        boards_in_block = min(config["stacks_y"], config["actual_boards"] - start_board_index)
                        if boards_in_block > 0:
                            block_depth = boards_in_block * oriented_dims[1]
                            blocks.append({
                                "position": (x_pos, y_pos, z_pos), "dimensions": (oriented_dims[0], block_depth, oriented_dims[2]),
                                "boards": boards_in_block, "type": "length_height_block"
                            })
                            
        elif orientation_idx in [3, 5]:
            for by in range(config["stacks_y"]):
                for bz in range(config["boards_per_stack"]):
                    x_pos, y_pos, z_pos = position[0], position[1] + by * oriented_dims[1], position[2] + bz * oriented_dims[2]
                    start_board_index = (by * config["boards_per_stack"] + bz) * config["stacks_x"]
                    if start_board_index < config["actual_boards"]:
                        boards_in_block = min(config["stacks_x"], config["actual_boards"] - start_board_index)
                        if boards_in_block > 0:
                            block_width = boards_in_block * oriented_dims[0]
                            blocks.append({
                                "position": (x_pos, y_pos, z_pos), "dimensions": (block_width, oriented_dims[1], oriented_dims[2]),
                                "boards": boards_in_block, "type": "width_height_block"
                            })
        return blocks

    def visualize_loading_subplot(self, row, col, results_data):
        """Create a single 3D visualization for subplot"""
        # (This function remains unchanged)
        traces = []
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8', '#F7DC6F']
        container_vertices = np.array([[0, 0, 0], [self.container["L"], 0, 0], [self.container["L"], self.container["W"], 0], [0, self.container["W"], 0], [0, 0, self.container["H"]], [self.container["L"], 0, self.container["H"]], [self.container["L"], self.container["W"], self.container["H"]], [0, self.container["W"], self.container["H"]]])
        container_edges = [[0, 1], [1, 2], [2, 3], [3, 0], [4, 5], [5, 6], [6, 7], [7, 4], [0, 4], [1, 5], [2, 6], [3, 7]]
        for edge in container_edges:
            start, end = edge
            traces.append(go.Scatter3d(x=[container_vertices[start][0], container_vertices[end][0]], y=[container_vertices[start][1], container_vertices[end][1]], z=[container_vertices[start][2], container_vertices[end][2]], mode='lines', line=dict(color='rgba(0,0,0,0.9)', width=4), showlegend=False, hoverinfo='skip'))
        for placement_idx, placement in enumerate(self.loaded_boards):
            config = placement["config"]
            board_type = placement["board_type"]
            orientation_idx = config["orientation_index"]
            color = colors[orientation_idx % len(colors)]
            blocks = self.get_block_structure(placement)
            for block_idx, block in enumerate(blocks):
                block_pos, block_dims = block["position"], block["dimensions"]
                vertices, faces = self.create_block_mesh(block_pos[0], block_pos[1], block_pos[2], block_dims[0], block_dims[1], block_dims[2])
                traces.append(go.Mesh3d(x=vertices[:, 0], y=vertices[:, 1], z=vertices[:, 2], i=[f[0] for f in faces], j=[f[1] for f in faces], k=[f[2] for f in faces], color=color, opacity=0.8, showscale=False, hovertemplate=f"Board Type: {board_type['id']}<br>Orientation: {config['orientation_name']}<br>Boards in Block: {block['boards']}<br>Block Dimensions: {block_dims[0]:.0f} x {block_dims[1]:.0f} x {block_dims[2]:.0f}<br>Position: ({block_pos[0]:.0f}, {block_pos[1]:.0f}, {block_pos[2]:.0f})", showlegend=False))
        return traces

    def compare_starting_orientations(self):
        """Run optimization for each starting orientation and create comparison"""
        # (This function remains unchanged)
        results = {}
        print("Running optimization for each starting orientation...")
        selected_indices = [0, 1] 
        results = {}
        for i in selected_indices:
            orientation_name = self.orientation_names[i]
            print(f"\nRunning optimization starting with: {orientation_name}")
            self.loaded_boards, self.residual_spaces = [], []
            total_loaded, loaded_boards = self.optimize_loading_with_forced_first_orientation(i)
            stats = self.calculate_statistics()
            results[i] = {'orientation_name': orientation_name, 'total_loaded': total_loaded, 'loaded_boards': loaded_boards.copy(), 'stats': stats, 'volume_utilization': stats['volume_utilization'], 'weight_utilization': stats['weight_utilization']}
            print(f"Result: {total_loaded:,} boards loaded ({stats['volume_utilization']:.1f}% volume, {stats['weight_utilization']:.1f}% weight)")
        return results

    def visualize_orientation_comparison(self, results):
        """Create subplots comparing different starting orientations."""
        # (This function remains unchanged, but I've included it for completeness)
        num_results = len(results)
        if num_results == 0:
            print("No results to visualize.")
            return go.Figure(), results
        cols = min(num_results, 2)
        rows = (num_results + cols - 1) // cols
        subplot_titles = [f"Solution {i+1}: {res['orientation_name']}" for i,res in enumerate(results.values())]
        fig = make_subplots(rows=rows, cols=cols, subplot_titles=subplot_titles, specs=[[{"type": "scene"}] * cols] * rows, horizontal_spacing=0.08, vertical_spacing=0.25)
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8', '#F7DC6F']
        all_orientations_used = set()
        for i, (orientation_idx, result) in enumerate(results.items()):
            row, col = (i // cols) + 1, (i % cols) + 1
            self.loaded_boards = result['loaded_boards']
            traces = self.visualize_loading_subplot_improved(row, col, result, colors)
            for placement in self.loaded_boards:
                all_orientations_used.add(placement["config"]["orientation_index"])
            for trace in traces:
                fig.add_trace(trace, row=row, col=col)
        current_annotations = list(fig.layout.annotations)
        fig.layout.annotations = [] 
        for i, (orientation_idx, result) in enumerate(results.items()):
            title_annotation = current_annotations[i]
            fig.add_annotation(title_annotation)
            annotation_text = f"Total Boards: {result['total_loaded']:,}<br>Volume Util: {result['volume_utilization']:.1f}%<br>Weight Util: {result['weight_utilization']:.1f}%"
            fig.add_annotation(text=annotation_text, xref="paper", yref="paper", x=title_annotation.x, y=title_annotation.y - 0.08, showarrow=False, font=dict(size=12, color="#333"), align="center", bgcolor="rgba(250, 250, 250, 0.85)", bordercolor="rgba(0, 0, 0, 0.2)", borderwidth=1, yshift=-10)
        for orientation_idx in sorted(all_orientations_used):
            fig.add_trace(go.Scatter3d(x=[None], y=[None], z=[None], mode='markers', marker=dict(color=colors[orientation_idx % len(colors)], size=10, symbol='square'), name=self.orientation_names[orientation_idx], showlegend=True, legendgroup="orientations"))
        container_specs = f"{self.container['Name']} Container: {self.container['L']}L x {self.container['W']}W x {self.container['H']}H mm | Max Weight: {self.container.get('max_weight', 'N/A'):,} kg"
        fig_height = 750 * rows
        fig.update_layout(title=dict(text=f"Container Loading Optimization<br>{container_specs}", x=0.5, y=0.97, font=dict(size=20)), height=fig_height, width=1800, showlegend=True, legend=dict(title="Orientations Used", x=1.0, y=0.6, bgcolor="rgba(255,255,255,0.9)", bordercolor="#ccc", borderwidth=1), margin=dict(l=50, r=250, t=120, b=50))
        scene_configs = {'xaxis': dict(title="Length (mm)"), 'yaxis': dict(title="Width (mm)"), 'zaxis': dict(title="Height (mm)"), 'aspectmode': "data", 'camera': dict(eye=dict(x=1.3, y=1.3, z=0.8))}
        for i in range(1, num_results + 1):
            scene_name = f'scene{i}' if i > 1 else 'scene'
            fig.update_layout(**{scene_name: scene_configs})
        return fig, results

    def visualize_loading_subplot_improved(self, row, col, results_data, colors):
        """Create a single 3D visualization for subplot with dark borders and detailed hover"""
        # (This function remains unchanged)
        traces = []
        container_vertices = np.array([[0, 0, 0], [self.container["L"], 0, 0], [self.container["L"], self.container["W"], 0], [0, self.container["W"], 0], [0, 0, self.container["H"]], [self.container["L"], 0, self.container["H"]], [self.container["L"], self.container["W"], self.container["H"]], [0, self.container["W"], self.container["H"]]])
        container_edges = [[0, 1], [1, 2], [2, 3], [3, 0], [4, 5], [5, 6], [6, 7], [7, 4], [0, 4], [1, 5], [2, 6], [3, 7]]
        for edge in container_edges:
            start, end = edge
            traces.append(go.Scatter3d(x=[container_vertices[start][0], container_vertices[end][0]], y=[container_vertices[start][1], container_vertices[end][1]], z=[container_vertices[start][2], container_vertices[end][2]], mode='lines', line=dict(color='rgba(0,0,0,0.8)', width=3), showlegend=False, hoverinfo='skip'))
        
        side_height_base = self.container.get('side_height')
        ground_clearance = self.container.get('ground_clearance', 0) 
        absolute_side_height = side_height_base + ground_clearance

        if absolute_side_height > 0 and absolute_side_height < self.container['H']:
            L, W = self.container['L'], self.container['W']
            # Define the corners of the side_height plane as a closed loop
            x_coords = [L, 0, 0, L, L]
            y_coords = [0, 0, W, W, 0]
            z_coords = [absolute_side_height, absolute_side_height, absolute_side_height, absolute_side_height, absolute_side_height]
            
            traces.append(go.Scatter3d(
                x=x_coords, y=y_coords, z=z_coords,
                mode='lines',
                line=dict(color='red', width=5, dash='dash'),
                name='Side Height Limit',
                hoverinfo='none', # No hover text for this line
                showlegend=False # Does not need its own legend entry
            ))
        for placement_idx, placement in enumerate(self.loaded_boards):
            config = placement["config"]
            board_type = placement["board_type"]
            orientation_idx = config["orientation_index"]
            color = colors[orientation_idx % len(colors)]
            blocks = self.get_block_structure(placement)
            for block_idx, block in enumerate(blocks):
                block_pos, block_dims = block["position"], block["dimensions"]
                vertices, faces = self.create_block_mesh(block_pos[0], block_pos[1], block_pos[2], block_dims[0], block_dims[1], block_dims[2])
                traces.append(go.Mesh3d(x=vertices[:, 0], y=vertices[:, 1], z=vertices[:, 2], i=[f[0] for f in faces], j=[f[1] for f in faces], k=[f[2] for f in faces], color=color, opacity=1.0, showscale=False, flatshading=True, lighting=dict(ambient=0.7, diffuse=0.8, specular=0.1), hovertemplate=f"<b>Board Type: {board_type['id']}</b><br>Count: {block['boards']} boards<br>Orientation: {config['orientation_name']}<br>Size: {block_dims[0]:.0f} × {block_dims[1]:.0f} × {block_dims[2]:.0f} mm<br>Position: ({block_pos[0]:.0f}, {block_pos[1]:.0f}, {block_pos[2]:.0f})<extra></extra>", showlegend=False))
                block_edges = [[0, 1], [1, 2], [2, 3], [3, 0], [4, 5], [5, 6], [6, 7], [7, 4], [0, 4], [1, 5], [2, 6], [3, 7]]
                for edge in block_edges:
                    start, end = edge
                    traces.append(go.Scatter3d(x=[vertices[start][0], vertices[end][0]], y=[vertices[start][1], vertices[end][1]], z=[vertices[start][2], vertices[end][2]], mode='lines', line=dict(color='black', width=4), showlegend=False, hoverinfo='skip'))
        return traces

def main_comparison():
    """Main function to run the orientation comparison"""
    boards = [
        {"id": "PL Board", "dims": (1829, 1220, 12), "quantity": 10000, "weight": 15.6}
    ]
    board_thickness = boards[0]["dims"][2]
    dynamic_ground_clearance = board_thickness * 7
    
    container = {"Name":'21MT', "L": 6710, "W": 2400, "H": 2400 - dynamic_ground_clearance, "max_weight": 23000, "side_height": 1500 - dynamic_ground_clearance, 'max_boards_stack': 50,'ground_clearance': dynamic_ground_clearance}

    optimizer = ContainerLoadingOptimizer(container, boards)
    results = optimizer.compare_starting_orientations()
    fig, results_data = optimizer.visualize_orientation_comparison(results)

    print("\n" + "="*80)
    print(" STARTING ORIENTATION COMPARISON RESULTS")
    print("="*80)
    sorted_results = sorted(results.items(), key=lambda x: x[1]['total_loaded'], reverse=True)
    print(f"\n📊 SUMMARY (Ranked by Total Boards Loaded):")
    print("-" * 80)
    for rank, (orientation_idx, result) in enumerate(sorted_results, 1):
        print(f"{rank}. Starting Orientation: {result['orientation_name']}")
        print(f"   Total Boards: {result['total_loaded']:,}")
        print(f"   Volume Utilization: {result['volume_utilization']:.2f}%")
        print(f"   Weight Utilization: {result['weight_utilization']:.2f}%")
        print()
    
    fig.show()

    return optimizer, results_data

if __name__ == "__main__":
    optimizer, results = main_comparison()
