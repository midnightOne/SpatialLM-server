import asyncio
import websockets
import numpy as np
import json
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def generate_room_point_cloud():
    """Generate a simple room point cloud for testing."""
    points = []
    
    # Room dimensions (in meters)
    width, length, height = 4, 6, 2.5
    
    # Floor
    for x in np.linspace(0, width, 20):
        for z in np.linspace(0, length, 30):
            points.append([x, 0, z])
    
    # Ceiling
    for x in np.linspace(0, width, 20):
        for z in np.linspace(0, length, 30):
            points.append([x, height, z])
    
    # Walls
    # Front wall
    for x in np.linspace(0, width, 20):
        for y in np.linspace(0, height, 10):
            points.append([x, y, 0])
    
    # Back wall
    for x in np.linspace(0, width, 20):
        for y in np.linspace(0, height, 10):
            points.append([x, y, length])
    
    # Left wall
    for z in np.linspace(0, length, 30):
        for y in np.linspace(0, height, 10):
            points.append([0, y, z])
    
    # Right wall
    for z in np.linspace(0, length, 30):
        for y in np.linspace(0, height, 10):
            points.append([width, y, z])
    
    # Add some random noise
    points = np.array(points)
    noise = np.random.normal(0, 0.01, points.shape)
    points = points + noise
    
    return points

async def test_server():
    try:
        # Generate room point cloud
        points = generate_room_point_cloud()
        logger.info(f"Generated point cloud with shape: {points.shape}")
        
        # Connect to server
        uri = "ws://localhost:8765"
        async with websockets.connect(uri) as websocket:
            logger.info("Connected to server")
            
            # Send point cloud data
            message = {
                "points": points.tolist()
            }
            await websocket.send(json.dumps(message))
            logger.info("Sent point cloud data")
            
            # Receive response
            response = await websocket.recv()
            logger.info("Received response:")
            print(json.loads(response))
    except Exception as e:
        logger.error(f"Error in test client: {str(e)}")
        raise

if __name__ == "__main__":
    asyncio.run(test_server()) 