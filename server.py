import asyncio
import websockets
import torch
import numpy as np
from transformers import AutoTokenizer, TextIteratorStreamer
from spatiallm import Layout, SpatialLMLlamaForCausalLM
import json
import logging
from torchsparse import SparseTensor
from torchsparse.utils.collate import sparse_collate
from spatiallm.pcd import Compose, load_o3d_pcd, get_points_and_colors, cleanup_pcd
from threading import Thread

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

class SpatialLMServer:
    def __init__(self, model_path="manycore-research/SpatialLM-Llama-1B"):
        logger.info("Initializing SpatialLM server...")
        # Initialize model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = SpatialLMLlamaForCausalLM.from_pretrained(model_path)
        self.model.to("cuda" if torch.cuda.is_available() else "cpu")
        self.model.eval()
        logger.info(f"Model loaded on device: {self.model.device}")
        
        if torch.cuda.is_available():
            self.model.set_point_backbone_dtype(torch.float32)
            
        # Initialize point cloud preprocessing
        self.grid_size = Layout.get_grid_size()
        self.num_bins = Layout.get_num_bins()
        self.transform = Compose([
            dict(type="PositiveShift"),
            dict(type="NormalizeColor"),
            dict(
                type="GridSample",
                grid_size=self.grid_size,
                hash_type="fnv",
                mode="test",
                keys=("coord", "color"),
                return_grid_coord=True,
                max_grid_coord=self.num_bins,
            ),
        ])

    def prepare_point_cloud(self, points):
        """Convert numpy array to tensor format expected by the model."""
        # Create dummy colors (zeros) since we don't have color information
        colors = np.zeros_like(points, dtype=np.uint8)
        
        # Prepare data dictionary for transform
        data_dict = {
            "name": "pcd",
            "coord": points.copy(),
            "color": colors.copy(),
        }
        
        # Apply transforms
        point_cloud = self.transform(data_dict)
        
        # Combine coordinates and features
        coord = point_cloud["grid_coord"]
        xyz = point_cloud["coord"]
        rgb = point_cloud["color"]
        point_cloud = np.concatenate([coord, xyz, rgb], axis=1)
        
        # Convert to tensor and add batch dimension
        return torch.as_tensor(np.stack([point_cloud], axis=0))

    def boxes_to_dict(self, boxes):
        """Convert bounding boxes to a JSON-serializable format."""
        result = []
        for box in boxes:
            box_dict = {
                "id": box["id"],
                "class": box["class"],
                "label": box["label"],
                "center": box["center"].tolist(),
                "rotation": box["rotation"].tolist(),
                "scale": box["scale"].tolist()
            }
            result.append(box_dict)
        return result

    async def process_point_cloud(self, point_cloud_data, websocket):
        try:
            logger.info(f"Processing point cloud with shape: {point_cloud_data.shape}")
            
            # Convert point cloud data to tensor
            point_cloud = self.prepare_point_cloud(point_cloud_data)
            
            # Prepare prompt with explicit format instructions
            prompt = """<|point_start|><|point_pad|><|point_end|>Detect walls, doors, windows, and furniture in the room.
Format each element as:
wall_0=Wall(ax,ay,az,bx,by,bz,height,thickness)
door_0=Door(wall_id,position_x,position_y,position_z,width,height)
window_0=Window(wall_id,position_x,position_y,position_z,width,height)
bbox_0=Bbox(class_name,position_x,position_y,position_z,angle_z,scale_x,scale_y,scale_z)

Generate a complete room layout with:
1. Walls forming the room boundary (4 walls)
2. One door on the back wall
3. Furniture boxes including:
   - One bed (large rectangular object)
   - Two nightstands (small rectangular objects near the bed)
   - One curtain (thin rectangular object on a wall)
   - One carpet (flat rectangular object on floor)
   - One wall decoration (thin rectangular object on wall)

Make sure to:
1. Start with the walls to define the room boundary
2. Add exactly one door on the back wall
3. Place furniture boxes with correct positions and dimensions
4. Include ALL furniture in the room
5. Continue generating until you have identified all furniture

Common furniture dimensions:
- Beds: ~2m long, ~1.5m wide, ~0.5m high
- Nightstands: ~0.5m wide, ~0.5m deep, ~0.5m high
- Curtains: ~2m high, ~0.1m thick
- Carpets: ~2m wide, ~1.5m long, very thin
- Wall decorations: ~1m wide, ~0.1m thick, ~0.8m high"""
            
            conversation = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ]
            input_ids = self.tokenizer.apply_chat_template(
                conversation, add_generation_prompt=True, return_tensors="pt"
            )
            input_ids = input_ids.to(self.model.device)

            # Set up streaming with a timeout
            streamer = TextIteratorStreamer(
                self.tokenizer, 
                timeout=20.0, 
                skip_prompt=True, 
                skip_special_tokens=True
            )

            # Generate layout with streaming
            generate_kwargs = dict(
                input_ids=input_ids,
                point_clouds=point_cloud,
                max_length=4096,  # Increased max length to allow for more objects
                do_sample=True,
                temperature=0.7,  # Slightly increased temperature for more diverse outputs
                top_p=0.95,
                top_k=10,
                num_beams=1,
                streamer=streamer,
                pad_token_id=self.tokenizer.eos_token_id,
                attention_mask=torch.ones_like(input_ids)
            )

            # Start generation in a separate thread
            thread = Thread(target=self.model.generate, kwargs=generate_kwargs)
            thread.start()

            # Stream the generated text with a timeout
            generated_text = ""
            try:
                for text in streamer:
                    if text.strip():  # Only send non-empty text
                        # Clean up the text by removing extra newlines and duplicates
                        text = text.strip()
                        if text not in generated_text:  # Avoid duplicates
                            # Validate the format
                            if any(text.startswith(prefix) for prefix in ["wall_", "door_", "window_", "bbox_"]):
                                # Check for duplicate doors
                                if text.startswith("door_"):
                                    if "door_" in generated_text:
                                        continue
                                
                                generated_text += text + "\n"
                                # Send intermediate results
                                response = {
                                    "type": "stream",
                                    "text": text,
                                    "layout_text": generated_text.strip()
                                }
                                await websocket.send(json.dumps(response, cls=NumpyEncoder))
                                
                                # Only stop if we have:
                                # 1. Exactly 4 walls
                                # 2. Exactly 1 door
                                # 3. At least 6 furniture boxes (bed, 2 nightstands, curtain, carpet, wall decoration)
                                # 4. At least 10 total objects
                                if (generated_text.count("wall_") == 4 and 
                                    generated_text.count("door_") == 1 and 
                                    generated_text.count("bbox_") >= 6 and 
                                    len(generated_text.split("\n")) >= 10):
                                    break
            except Exception as e:
                logger.error(f"Error during streaming: {str(e)}")
                # If streaming fails, try to process what we have
                if generated_text:
                    layout = Layout(generated_text.strip())
                    boxes = layout.to_boxes()
                    response = {
                        "type": "final",
                        "boxes": self.boxes_to_dict(boxes),
                        "layout_text": generated_text.strip()
                    }
                    await websocket.send(json.dumps(response, cls=NumpyEncoder))
                    return boxes, generated_text
                raise

            # After generation is complete, process the final layout
            layout = Layout(generated_text.strip())
            boxes = layout.to_boxes()
            logger.info(f"Generated {len(boxes)} bounding boxes")

            # Send final results
            response = {
                "type": "final",
                "boxes": self.boxes_to_dict(boxes),
                "layout_text": generated_text.strip()
            }
            await websocket.send(json.dumps(response, cls=NumpyEncoder))

            return boxes, generated_text
        except Exception as e:
            logger.error(f"Error processing point cloud: {str(e)}")
            raise

    async def handle_connection(self, websocket):
        try:
            async for message in websocket:
                logger.info("Received message from client")
                # Expect point cloud data as JSON
                data = json.loads(message)
                point_cloud = np.array(data["points"])
                logger.info(f"Received point cloud with shape: {point_cloud.shape}")
                
                # Process point cloud and get bounding boxes
                boxes, generated_text = await self.process_point_cloud(point_cloud, websocket)
                
        except Exception as e:
            logger.error(f"Error handling connection: {str(e)}")
            raise

    async def start_server(self, host="localhost", port=8765):
        # Increase max message size to 16MB
        async with websockets.serve(
            self.handle_connection, 
            host, 
            port,
            max_size=16_777_216,  # 16MB
            max_queue=32
        ):
            logger.info(f"Server started on ws://{host}:{port}")
            await asyncio.Future()  # run forever

if __name__ == "__main__":
    server = SpatialLMServer()
    asyncio.run(server.start_server())