import pandas as pd
from PIL import Image
import torch
from tqdm import tqdm
import os, tarfile

from transformers import CLIPModel, CLIPProcessor

configs = [
    {
        "MODEL": "laion/CLIP-ViT-bigG-14-laion2B-39B-b160k",
        "FILE_NAME": "openclip_vit_bigg_14"
    },
    {
        "MODEL": "laion/CLIP-ViT-H-14-laion2B-s32B-b79K",
        "FILE_NAME": "openclip_vit_h_14"
    },
    {
        "MODEL": "laion/CLIP-ViT-L-14-laion2B-s32B-b82K",
        "FILE_NAME": "openclip_vit_l_14"
    }
]

BATCH_SIZE = 400
images_folder = "datasets/LOGO/images/"
#LOGO parquet downloaded from https://huggingface.co/datasets/ChristophSchuhmann/aesthetic-logo-ratings
#images downloaded with img2dataset - https://github.com/rom1504/img2dataset
# (img2dataset --input_format=parquet --url_col=url --output_format=webdataset --output_folder=datasets/LOGO-images  --image_size=512 --url_list=datasets/LOGO/logos_ratings.parquet)
#img2dataset is missing indexes, so using url as reference

#df = pd.read_parquet('parquets/prepared_hord_diffusion_dataset.parquet')

logo_ratings_df = pd.read_parquet("datasets/LOGO/logos_ratings.parquet")


folder = os.fsencode(images_folder)


for config in configs:
    MODEL = config["MODEL"]
    print ("starting model "+MODEL)
    FILE_NAME = config["FILE_NAME"]
    embedding_file_path = f"embeddings/logo_{FILE_NAME}.parquet"
    if os.path.exists(embedding_file_path):
        result_df = pd.read_parquet(embedding_file_path)
    else:
        result_df = pd.DataFrame(columns=["image_url", "pooled_output", "projected_embedding", "professionalism_average","preference_average","number_of_raters"])

    model = CLIPModel.from_pretrained(MODEL)
    vision_model = model.vision_model
    visual_projection = model.visual_projection
    vision_model.to("cuda")
    visual_projection.to("cuda")
    del model
    processor = CLIPProcessor.from_pretrained(MODEL)

    for file in os.listdir(folder):
        filename = os.fsdecode(file)
        if filename.endswith(".parquet"): 
            current_tar_df = pd.read_parquet(images_folder+filename)
            current_tar = tarfile.open(f"{images_folder+filename.split('.')[0]}.tar")

            for pos in tqdm(range(0,len(current_tar_df), BATCH_SIZE)):
                name_batch = current_tar_df[pos:pos+BATCH_SIZE]
                pil_images = []
                existing_images_df =  pd.DataFrame(columns=["logo_key", "image_url", "professionalism_average","preference_average","number_of_raters"])
                
                for ix, item in name_batch.iterrows():
                    try:
                        member = current_tar.extractfile(f"{str(item['key'])}.jpg")
                        try:
                            logo_item = logo_ratings_df[logo_ratings_df['url']==item['url']]
                            img = Image.open(member)
                            img.load()
                            # print ("loaded image")
                            pil_images.append(img)
                            # print ("added to pil images")
                            temp_df =  pd.DataFrame({
                                "logo_key": logo_item.key.values[0],
                                "image_url": item['url'],
                                "professionalism_average": logo_item.proffesionalism_average.values[0],
                                "preference_average":logo_item.preference_average.values[0],
                                "number_of_raters": logo_item.number_of_raters.values[0]
                                }, index = [0])
                            # print ("made temp")
                            existing_images_df = pd.concat ([existing_images_df,temp_df])
                            # print (f"added to existing images len={len(existing_images_df)}")
                        except Exception as e:
                            print ("Problem reading image")
                            print (e)
                            exit(0)
                    except Exception as e:
                        #print ("Problem reading tarfile")
                        #print (e)
                        pass
                print (f"finished collecting images, batch size = {len(pil_images)}")
                if pil_images:
                    inputs = processor(images=pil_images, return_tensors="pt").to("cuda")
                    ratings_row =  logo_ratings_df.merge(name_batch, on=['url'])
                    with torch.no_grad():
                        vision_output = vision_model(**inputs)
                        pooled_output = vision_output.pooler_output
                        projected_embedding = visual_projection(pooled_output)
                        # print (f"pil images len {len(pil_images)}")
                        # print (f"exsiting images len {len(existing_images_df)}")
                        # print (f"porjected output len {len(pooled_output.cpu().detach().numpy())}")
                        # print (f"porjeprofessional len {len(existing_images_df['preference_average'])}")
                        # exit(0)
                        pd_res = pd.DataFrame({
                            "image_url": list(existing_images_df['image_url']), 
                            "pooled_output": list(pooled_output.cpu().detach().numpy()),
                            "projected_embedding": list(projected_embedding.cpu().detach().numpy()),
                            "professionalism_average": list(existing_images_df['professionalism_average']),
                            "preference_average":list(existing_images_df['preference_average']),
                            "number_of_raters": list(existing_images_df['number_of_raters'])
                            })
                        



                        result_df = pd.concat([result_df, pd_res], ignore_index=True)
                    result_df.to_parquet(embedding_file_path)


# missing_image_names = ava_df[~ava_df["image_name"].isin(result_df["image_name"])]["image_name"].unique()

#print(f"Missing {len(missing_image_names)} embeddings...")



