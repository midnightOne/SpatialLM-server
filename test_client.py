import asyncio
import websockets
import numpy as np
import json
import logging
import argparse
from spatiallm.pcd import load_o3d_pcd, get_points_and_colors, cleanup_pcd

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def generate_room_point_cloud(num_points=2200):
    """Generate a point cloud representing a room with some furniture."""
    # Room dimensions (in meters)
    room_width = 4.0
    room_depth = 6.0
    room_height = 2.5
    
    # Generate points for the room
    points = []
    
    # Floor
    floor_points = np.random.rand(num_points // 4, 3)
    floor_points[:, 0] *= room_width
    floor_points[:, 1] *= room_depth
    floor_points[:, 2] = 0
    points.append(floor_points)
    
    # Walls
    # Front wall
    wall_points = np.random.rand(num_points // 8, 3)
    wall_points[:, 0] *= room_width
    wall_points[:, 1] = 0
    wall_points[:, 2] *= room_height
    points.append(wall_points)
    
    # Back wall
    wall_points = np.random.rand(num_points // 8, 3)
    wall_points[:, 0] *= room_width
    wall_points[:, 1] = room_depth
    wall_points[:, 2] *= room_height
    points.append(wall_points)
    
    # Left wall
    wall_points = np.random.rand(num_points // 8, 3)
    wall_points[:, 0] = 0
    wall_points[:, 1] *= room_depth
    wall_points[:, 2] *= room_height
    points.append(wall_points)
    
    # Right wall
    wall_points = np.random.rand(num_points // 8, 3)
    wall_points[:, 0] = room_width
    wall_points[:, 1] *= room_depth
    wall_points[:, 2] *= room_height
    points.append(wall_points)
    
    # Ceiling
    ceiling_points = np.random.rand(num_points // 4, 3)
    ceiling_points[:, 0] *= room_width
    ceiling_points[:, 1] *= room_depth
    ceiling_points[:, 2] = room_height
    points.append(ceiling_points)
    
    # Combine all points
    points = np.concatenate(points, axis=0)
    
    # Add some random noise
    noise = np.random.normal(0, 0.01, points.shape)
    points += noise
    
    return points

def load_point_cloud(file_path):
    """Load a point cloud from a file."""
    logger.info(f"Loading point cloud from {file_path}")
    pcd = load_o3d_pcd(file_path)
    
    # Clean up and downsample the point cloud
    pcd = cleanup_pcd(pcd, voxel_size=0.02)  # 2cm voxel size for downsampling
    points, colors = get_points_and_colors(pcd)
    
    # Center and normalize the point cloud
    points = points - np.mean(points, axis=0)
    scale = np.max(np.abs(points))
    points = points / scale * 2.0  # Scale to [-2, 2] range
    
    logger.info(f"Loaded point cloud with shape: {points.shape}")
    return points

async def test_server(point_cloud_file=None):
    try:
        # Generate or load point cloud
        if point_cloud_file:
            points = load_point_cloud(point_cloud_file)
        else:
            points = generate_room_point_cloud()
            
        logger.info(f"Point cloud shape: {points.shape}")
        
        # Connect to the server with increased message size
        async with websockets.connect(
            "ws://localhost:8765",
            max_size=16_777_216  # 16MB
        ) as websocket:
            logger.info("Connected to server")
            
            # Send point cloud data
            data = {"points": points.tolist()}
            await websocket.send(json.dumps(data))
            logger.info("Sent point cloud data")
            
            # Receive streaming responses with timeout
            start_time = asyncio.get_event_loop().time()
            timeout = 30  # 30 seconds timeout
            
            while True:
                try:
                    # Check for timeout
                    if asyncio.get_event_loop().time() - start_time > timeout:
                        logger.error("Timeout waiting for server response")
                        break
                        
                    response = await asyncio.wait_for(websocket.recv(), timeout=5.0)
                    response_data = json.loads(response)
                    
                    if response_data["type"] == "stream":
                        if response_data["text"].strip():  # Only log non-empty text
                            logger.info(f"Streaming text: {response_data['text']}")
                    elif response_data["type"] == "final":
                        logger.info("Received final response:")
                        # Print layout text with proper formatting
                        layout_text = response_data["layout_text"]
                        logger.info("Layout text:")
                        for line in layout_text.split("\n"):
                            if line.strip():
                                logger.info(f"  {line}")
                        
                        # Print box information
                        boxes = response_data["boxes"]
                        logger.info(f"Number of boxes: {len(boxes)}")
                        if boxes:
                            logger.info("Box details:")
                            # Group boxes by type
                            boxes_by_type = {}
                            for box in boxes:
                                box_type = box['class']
                                if box_type not in boxes_by_type:
                                    boxes_by_type[box_type] = []
                                boxes_by_type[box_type].append(box)
                            
                            # Print summary by type
                            for box_type, type_boxes in boxes_by_type.items():
                                logger.info(f"\n  {box_type.title()}s ({len(type_boxes)}):")
                                for box in type_boxes:
                                    logger.info(f"    {box['class']} {box['id']}:")
                                    logger.info(f"      Center: {box['center']}")
                                    logger.info(f"      Rotation: {box['rotation']}")
                                    logger.info(f"      Scale: {box['scale']}")
                                    
                                    # Calculate dimensions
                                    dims = [abs(s) for s in box['scale']]
                                    logger.info(f"      Dimensions: {dims}")
                                    
                                    # For walls, show orientation
                                    if box['class'] == 'wall':
                                        if dims[0] > dims[1] and dims[0] > dims[2]:
                                            orientation = "horizontal"
                                        elif dims[1] > dims[0] and dims[1] > dims[2]:
                                            orientation = "vertical"
                                        else:
                                            orientation = "unknown"
                                        logger.info(f"      Orientation: {orientation}")
                        break
                        
                except asyncio.TimeoutError:
                    logger.warning("Timeout waiting for message, retrying...")
                    continue
                    
    except Exception as e:
        logger.error(f"Error in test client: {str(e)}")
        raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test SpatialLM server")
    parser.add_argument("--point-cloud", type=str, help="Path to point cloud file to load")
    args = parser.parse_args()
    
    asyncio.run(test_server(args.point_cloud)) 