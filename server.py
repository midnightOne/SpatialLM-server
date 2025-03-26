import asyncio
import websockets
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from spatiallm import Layout
import json
import logging
from torchsparse import SparseTensor
from torchsparse.utils.collate import sparse_collate_fn
from spatiallm.pcd import Compose

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
        self.model = AutoModelForCausalLM.from_pretrained(model_path)
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

    async def process_point_cloud(self, point_cloud_data):
        try:
            logger.info(f"Processing point cloud with shape: {point_cloud_data.shape}")
            
            # Convert point cloud data to tensor
            point_cloud = self.prepare_point_cloud(point_cloud_data)
            
            # Prepare prompt
            prompt = "<|point_start|><|point_pad|><|point_end|>Detect walls, doors, windows, boxes."
            conversation = [{"role": "user", "content": prompt}]
            input_ids = self.tokenizer.apply_chat_template(
                conversation, add_generation_prompt=True, return_tensors="pt"
            )
            input_ids = input_ids.to(self.model.device)

            # Generate layout
            with torch.no_grad():
                outputs = self.model.generate(
                    input_ids,
                    point_clouds=point_cloud,
                    max_length=4096,
                    do_sample=True,
                    temperature=0.6,
                    top_p=0.95,
                    top_k=10,
                    num_beams=1,
                    return_dict_in_generate=True,
                    output_scores=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    attention_mask=torch.ones_like(input_ids)
                )
                
                # Process outputs into layout
                generated_text = self.tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)
                logger.info(f"Generated text: {generated_text}")
                layout = Layout(generated_text)
                
                # Convert layout to bounding boxes
                boxes = layout.to_boxes()
                logger.info(f"Generated {len(boxes)} bounding boxes")
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
                boxes, generated_text = await self.process_point_cloud(point_cloud)
                
                # Convert boxes to JSON-serializable format
                boxes_dict = self.boxes_to_dict(boxes)
                
                # Send results back
                response = {
                    "boxes": boxes_dict,
                    "layout_text": generated_text  # Include the layout text for debugging
                }
                logger.info("Sending response back to client")
                await websocket.send(json.dumps(response, cls=NumpyEncoder))
                logger.info("Response sent successfully")
        except Exception as e:
            logger.error(f"Error handling connection: {str(e)}")
            raise

    async def start_server(self, host="localhost", port=8765):
        async with websockets.serve(self.handle_connection, host, port):
            logger.info(f"Server started on ws://{host}:{port}")
            await asyncio.Future()  # run forever

if __name__ == "__main__":
    server = SpatialLMServer()
    asyncio.run(server.start_server())