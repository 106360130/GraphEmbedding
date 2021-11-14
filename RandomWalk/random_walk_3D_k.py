import numpy as np
import matplotlib.pyplot as plt

# Define parameters for the walk 
dims = 3 
n_runs = 10 
step_n = 1000 
step_set = [-1, 0 ,1] 
runs = np.arange(n_runs) 
step_shape = (step_n,dims) 

# Plot 
fig = plt.figure(figsize=(10,10),dpi=250) 
ax = fig.add_subplot(111, projection='3d') 
ax.grid(False) 
ax.xaxis.pane.fill = ax.yaxis.pane.fill = ax.zaxis.pane.fill = False 
ax.set_xlabel('X') 
ax.set_ylabel('Y') 
ax.set_zlabel('Z') 

colors = ['blue', 'red', 'green', 'black']
for i, col in zip(runs, colors): 
    # Simulate steps in 3D 
    origin = np.random.randint(low=-10,high=10,size=(1,dims)) 
    steps = np.random.choice(a=step_set, size=step_shape) 
    path = np.concatenate([origin, steps]).cumsum(0) 
    start = path[:1] 
    stop = path[-1:] 
    
    # Plot the path 
    ax.scatter3D(path[:,0], path[:,1], path[:,2], c=col,alpha=0.15,s=1); 
    ax.plot3D(path[:,0], path[:,1], path[:,2], c=col, alpha=0.25,lw=0.25) 
    ax.plot3D(start[:,0], start[:,1], start[:,2], c=col, marker='+') 
    ax.plot3D(stop[:,0], stop[:,1], stop[:,2], c=col, marker='o'); 
    plt.title('3D Random Walk - Multiple runs')


plt.show()

