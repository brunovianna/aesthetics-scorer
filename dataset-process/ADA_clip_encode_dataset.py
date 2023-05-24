import pandas as pd
from PIL import Image
import torch
from tqdm import tqdm
import os

from transformers import CLIPModel, CLIPProcessor

configs = [
#    {
#        "MODEL": "laion/CLIP-ViT-bigG-14-laion2B-39B-b160k",
#        "FILE_NAME": "openclip_vit_bigg_14"
#    },
#    {
#        "MODEL": "laion/CLIP-ViT-H-14-laion2B-s32B-b79K",
#        "FILE_NAME": "openclip_vit_h_14"
#    },
    {
        "MODEL": "laion/CLIP-ViT-L-14-laion2B-s32B-b82K",
        "FILE_NAME": "openclip_vit_l_14"
    }
]

BATCH_SIZE = 400

#df = pd.read_parquet('parquets/prepared_hord_diffusion_dataset.parquet')


ava_df = pd.read_table ("dataset/AVA.txt", sep=" ",names=["index", "image_name", "1","2","3","4","5","6","7","8","9","10","tag_1", "tag_2", "challenge_id"])
 #images id is column 2.
#Column 1: Index
#Column 2: Image ID 
#Columns 3 - 12: Counts of aesthetics ratings on a scale of 1-10. Column 3 has counts of ratings of 1 and column 12 has counts of ratings of 10.
#Columns 13 - 14: 
# Semantic tag IDs. There are 66 IDs ranging from 1 to 66. The file tags.txt contains the textual tag corresponding to the numerical
# id. Each image has between 0 and 2 tags. Images with less than 2 tags have
#a "0" in place of the missing tag(s).

#Column 15: Challenge ID. The file challenges.txt contains the name of  the challenge corresponding to each ID.

for config in configs:
    MODEL = config["MODEL"]
    print ("starting model "+MODEL)
    FILE_NAME = config["FILE_NAME"]
    embedding_file_path = f"embeddings/ada_{FILE_NAME}.parquet"
    dataset_image_path = "dataset/images/"
    if os.path.exists(embedding_file_path):
        result_df = pd.read_parquet(embedding_file_path)
    else:
        result_df = pd.DataFrame(columns=["image_name", "pooled_output", "projected_embedding"])

    missing_image_names = ava_df[~ava_df["image_name"].isin(result_df["image_name"])]["image_name"].unique()

    print(f"Missing {len(missing_image_names)} embeddings...")


    model = CLIPModel.from_pretrained(MODEL)
    vision_model = model.vision_model
    visual_projection = model.visual_projection
    vision_model.to("cuda")
    visual_projection.to("cuda")
    del model
    processor = CLIPProcessor.from_pretrained(MODEL)

    for pos in tqdm(range(0, len(missing_image_names), BATCH_SIZE)):
        name_batch = missing_image_names[pos:pos+BATCH_SIZE]
        pil_images = []
        existing_images = []
        for id in name_batch:
            image_path = f"{dataset_image_path}{id}.jpg"
            try:
                # try to open that image and load the data
                i = Image.open(image_path)
                try: 
                    i.load() #making sure it is not truncated
                    pil_images.append(i)
                    existing_images.append(id)
                except Exception as e:
                    print (f"image {id} truncated")
            except Exception as e:
                print (f"image {id} not found")

        if len(pil_images) > 0:
            inputs = processor(images=pil_images, return_tensors="pt").to("cuda")
            with torch.no_grad():
                vision_output = vision_model(**inputs)
                pooled_output = vision_output.pooler_output
                projected_embedding = visual_projection(pooled_output)
                pd_res = pd.DataFrame({
               	    "image_name": existing_images, 
                    "pooled_output": list(pooled_output.cpu().detach().numpy()),
                    "projected_embedding": list(projected_embedding.cpu().detach().numpy()),
                })

                result_df = pd.concat([result_df, pd_res], ignore_index=True)

            #   try so save every batch
            result_df.to_parquet(embedding_file_path)








