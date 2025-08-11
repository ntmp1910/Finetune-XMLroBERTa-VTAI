import openai
from olmocr.anchor import get_anchor_text
from olmocr.renderpdf import render_pdf_to_base64png

from pypdf import PdfReader

import asyncio
import tempfile

from utils.s3_utils import get_s3_bytes_with_backoff
from utils.image_utils import convert_image_to_pdf_bytes, is_jpeg, is_png

import boto3
from botocore.exceptions import ClientError

import logging
import json

import os

from tqdm.asyncio import tqdm_asyncio

pdf_s3 = boto3.client("s3")

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.propagate = False



client = openai.Client(base_url="http://10.254.138.189:8102/v1", api_key="EMPTY")


def build_finetuning_prompt(base_text: str) -> str:
    return (
        f"Below is the image of one page of a document, as well as some raw textual content that was previously extracted for it. "
        f"Just return the plain text representation of this document as if you were reading it naturally.\n"
        f"Do not hallucinate.\n"
        f"RAW_TEXT_START\n{base_text}\nRAW_TEXT_END"
    )


async def process_file(pdf_orig_path: str, file_out_path: str):
    import functools
    with tempfile.NamedTemporaryFile("wb+", suffix=".pdf") as tf:
        try:
            data = await asyncio.to_thread(lambda: get_s3_bytes_with_backoff(pdf_s3, pdf_orig_path))
            tf.write(data)
            tf.flush()
        except ClientError as ex:
            if ex.response["Error"]["Code"] == "NoSuchKey":
                logger.info(f"S3 File Not found, skipping it completely {pdf_orig_path}")
                return None
            else:
                raise

        if is_png(tf.name) or is_jpeg(tf.name):
            logger.info(f"Converting {pdf_orig_path} from image to PDF format...")
            tf.seek(0)
            tf.write(convert_image_to_pdf_bytes(tf.name))
            tf.flush()

        try:
            reader = await asyncio.to_thread(PdfReader, tf.name)
            num_pages = await asyncio.to_thread(reader.get_num_pages)
        except:
            logger.exception(f"Could not count number of pages for {pdf_orig_path}, aborting document")
            return None

        logger.info(f"Got {num_pages} pages to do for {pdf_orig_path}")

        try:

            fout = await asyncio.to_thread(open, file_out_path, "w", encoding="utf-8")
            for page_num in range(1, num_pages + 1):
                image_base64 = await asyncio.to_thread(render_pdf_to_base64png, pdf_orig_path, page_num,
                                                       target_longest_image_dim=1024)
                anchor_text = await asyncio.to_thread(get_anchor_text, pdf_orig_path, page_num, "pdfreport", 4096)
                prompt = build_finetuning_prompt(anchor_text)


                def call_ocr():
                    return client.chat.completions.create(
                        model="RolmOCR",
                        messages=[
                            {
                                "role": "user",
                                "content": [
                                    {"type": "text", "text": prompt},
                                    {"type": "image_url",
                                     "image_url": {"url": f"data:image/png;base64,{image_base64}"}},
                                ],
                            }
                        ],
                        max_tokens=3000,
                        temperature=0.8,
                    ).choices[0].message.content

                response = await asyncio.to_thread(call_ocr)
                text = response
                if text is None:
                    continue
                text = text.replace("\n\n", "\n")
                await asyncio.to_thread(fout.write, text + "\n")
            await asyncio.to_thread(fout.close)
        except Exception as e:
            print(e)
            logger.exception(f"Exception in process_pdf for {pdf_orig_path}: {e}")
            return None


async def process_folder():
    folder_in_path = "/storage-nlp/nlp/dungdx4/tmp/Vietnamese/phuong/batch2/GIA ĐÌNH/PHONG THỦY"
    folder_out_path = "/storage-nlp/nlp/dungdx4/tmp/Vietnamese/phuong/Result/batch2/Ketqua batch2/GIA ĐÌNH/PHONG THỦY"

    files = os.listdir(folder_in_path)
    files_out = os.listdir(folder_out_path)


    semaphore = asyncio.Semaphore(21)

    async def process_with_semaphore(file_in_path, file_out_path):
        async with semaphore:
            return await process_file(file_in_path, file_out_path)

    tasks = []
    for file in files:
        print(f"Preparing to process: {file}")

        file_in_path = os.path.join(folder_in_path, file)
        file_tgt = file[:-3] + "txt"
        file_out_path = os.path.join(folder_out_path, file_tgt)


        if file_tgt in files_out:
            print(f"Skipping {file} - already processed")
            continue


        task = asyncio.create_task(process_with_semaphore(file_in_path, file_out_path))
        tasks.append(task)

    if tasks:
        print(f"Processing {len(tasks)} files concurrently...")
        for f in tqdm_asyncio.as_completed(tasks, total=len(tasks), desc="Progress"):
            await f
        results = await asyncio.gather(*tasks, return_exceptions=True)
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                print(f"Error processing file {files[i]}: {result}")
            else:
                print(f"Successfully processed: {files[i]}")
    else:
        print("No files to process")


if __name__ == "__main__":
    asyncio.run(process_folder())

