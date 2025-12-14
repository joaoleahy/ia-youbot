from controller import Supervisor
import random
import math
import sys

supervisor = Supervisor()
timestep = int(supervisor.getBasicTimeStep())
root_children = supervisor.getRoot().getField("children")

try:
    args = supervisor.getControllerArguments()
except AttributeError:
    args = sys.argv[1:]

if len(args) >= 5:
    n_objects = int(args[0])
    x_min = float(args[1])
    x_max = float(args[2])
    y_min = float(args[3])
    y_max = float(args[4])
else:
    n_objects = 15
    x_min, x_max = -3, 1.75
    y_min, y_max = -1, 1

#print(f"Spawner: {n_objects} objects in X[{x_min},{x_max}] Y[{y_min},{y_max}]")

def get_existing_obstacles():
    """Extract positions of WoodenBoxes and PlasticFruitBoxes from the world."""
    obstacles = []
    n = root_children.getCount()
    for i in range(n):
        node = root_children.getMFNode(i)
        type_name = node.getTypeName()
        
        if type_name in ["WoodenBox", "PlasticFruitBox"]:
            trans_field = node.getField("translation")
            size_field = node.getField("size")
            
            if trans_field:
                pos = trans_field.getSFVec3f()
                if size_field:
                    size = size_field.getSFVec3f()
                    radius = math.sqrt(size[0]**2 + size[1]**2) / 2
                else:
                    radius = 0.3
                
                obstacles.append({
                    'x': pos[0],
                    'y': pos[1],
                    'radius': radius + 0.1
                })
                #print(f"Found obstacle at ({pos[0]:.2f}, {pos[1]:.2f}) with radius {radius:.2f}")
    
    return obstacles

def delete_existing_objects():
    """Remove previously spawned objects."""
    n = root_children.getCount()
    for i in reversed(range(n)):
        node = root_children.getMFNode(i)
        name_field = node.getField("name")
        if name_field:
            node_name = name_field.getSFString()
            if node_name.startswith(("cube_", "cylinder_", "sphere_")):
                node.remove()

delete_existing_objects()

size = 0.03
mass = 0.03
min_dist = size * 2.5

existing_obstacles = get_existing_obstacles()

positions = []
colors = [
    (0, 1, 0),  # green
    (1, 0, 0),  # red
    (0, 0, 1),  # blue
]
shapes = ["cube"]

def random_pos():
    """Generate random position on the floor."""
    return (
        random.uniform(x_min, x_max),
        random.uniform(y_min, y_max),
        size / 2 + 0.001
    )

def is_far_enough(pos):
    """Check if position is far from both spawned objects and existing obstacles."""
    for q in positions:
        dx, dy = pos[0] - q[0], pos[1] - q[1]
        if math.hypot(dx, dy) < min_dist:
            return False
    
    for obs in existing_obstacles:
        dx, dy = pos[0] - obs['x'], pos[1] - obs['y']
        if math.hypot(dx, dy) < (obs['radius'] + size):
            return False
    
    return True

spawned_count = 0
failed_spawns = 0
max_attempts = 100

for i in range(n_objects):
    tries = 0
    success = False
    
    while tries < max_attempts:
        tries += 1
        pos = random_pos()
        if is_far_enough(pos):
            success = True
            break
    
    if not success:
        print(f"Warning: Could not find valid position for object {i} after {max_attempts} attempts")
        failed_spawns += 1
        continue
    
    positions.append(pos)
    shape_type = random.choice(shapes)
    color = random.choice(colors)
    
    if shape_type == "cube":
        geometry = f"Box {{ size {size} {size} {size} }}"
        bounding = f"Box {{ size {size} {size} {size} }}"
    
    # Node string
    node_string = f"""
    Solid {{
      translation {pos[0]} {pos[1]} {pos[2]}
      name "{shape_type}_{i}"
      children [
        Shape {{
          appearance PBRAppearance {{
            baseColor {color[0]} {color[1]} {color[2]}
            roughness 0.8
            metalness 0
          }}
          geometry {geometry}
        }}
      ]
      boundingObject {bounding}
      physics Physics {{
        density -1
        mass {mass}
      }}
      recognitionColors [ {color[0]} {color[1]} {color[2]} ]
    }}
    """
    root_children.importMFNodeFromString(-1, node_string)
    spawned_count += 1

print(f"Spawn complete. The supervisor has spawned {spawned_count}/{n_objects} objects ({failed_spawns} failed).")
#print("Initializing physics...")

for _ in range(20):
    supervisor.step(timestep)

#print("Supervisor ready.")

#while supervisor.step(timestep) != -1:
    #pass