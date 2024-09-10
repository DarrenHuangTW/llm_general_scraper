import os
import time
import re
import json
from typing import List, Dict, Type
from firecrawl import FirecrawlApp
import pandas as pd
from pydantic import BaseModel, create_model
import tiktoken

from dotenv import load_dotenv

from openai import OpenAI

load_dotenv()

def html_to_markdown_with_readability(url_input):
    app = FirecrawlApp(api_key="fc-303c4042c26340ba9afcddf153f57c64")
    scrape_result = app.scrape_url(url_input, params={'formats': ['markdown','screenshot']})
    markdown = scrape_result['markdown']
    try:
        screenshot_url = scrape_result['screenshot']
    except KeyError:
        screenshot_url = None
    metadata = scrape_result['metadata']
    title = metadata.get('title')
    status_code = metadata.get('statusCode')
    
    return markdown, screenshot_url, title, status_code
    

def save_raw_data(raw_data, timestamp, output_folder='output'):
    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)
    
    # Save the raw markdown data with timestamp in filename
    raw_output_path = os.path.join(output_folder, f'rawData_{timestamp}.md')
    with open(raw_output_path, 'w', encoding='utf-8') as f:
        f.write(raw_data)
    print(f"Raw data saved to {raw_output_path}")
    return raw_output_path


def create_dynamic_listing_model(field_names: List[str]) -> Type[BaseModel]:
    """
    Dynamically creates a Pydantic model based on provided fields.
    field_name is a list of names of the fields to extract from the markdown.
    """
    # Create field definitions using aliases for Field parameters
    field_definitions = {field: (str, ...) for field in field_names}
    # Dynamically create the model with all field
    return create_model('DynamicListingModel', **field_definitions)



def create_listings_container_model(listing_model: Type[BaseModel]) -> Type[BaseModel]:
    """
    Create a container model that holds a list of the given listing model.
    """
    return create_model('DynamicListingsContainer', listings=(List[listing_model], ...))


def trim_to_token_limit(text, model, max_tokens=200000):
    encoder = tiktoken.encoding_for_model(model)
    tokens = encoder.encode(text)
    if len(tokens) > max_tokens:
        trimmed_text = encoder.decode(tokens[:max_tokens])
        return trimmed_text
    return text


def format_data(data, DynamicListingsContainer, model_used):

    client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

    system_message = """You are an intelligent text extraction and conversion assistant. Your task is to extract structured information 
                        from the given text and convert it into a pure JSON format. The JSON should contain only the structured data extracted from the text, 
                        with no additional commentary, explanations, or extraneous information. 
                        You could encounter cases where you can't find the data of the fields you have to extract or the data will be in a foreign language.
                        Please process the following text and provide the output in pure JSON format with no words before or after the JSON:"""

    user_message = f"Extract the following information from the provided text:\nPage content:\n\n{data}"

    completion = client.beta.chat.completions.parse(
        model=model_used,
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message},
        ],
        response_format=DynamicListingsContainer
    )

    prompt_tokens = completion.usage.prompt_tokens
    completion_tokens = completion.usage.completion_tokens

    return completion.choices[0].message.parsed, prompt_tokens, completion_tokens


def save_formatted_data(formatted_data, timestamp, output_folder='output'):
    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)
    
    # Prepare formatted data as a dictionary
    formatted_data_dict = formatted_data.dict() if hasattr(formatted_data, 'dict') else formatted_data

    # Save the formatted data as JSON with timestamp in filename
    json_output_path = os.path.join(output_folder, f'sorted_data_{timestamp}.json')
    with open(json_output_path, 'w', encoding='utf-8') as f:
        json.dump(formatted_data_dict, f, indent=4)
    print(f"Formatted data saved to JSON at {json_output_path}")

    # Prepare data for DataFrame
    if isinstance(formatted_data_dict, dict):
        # If the data is a dictionary containing lists, assume these lists are records
        data_for_df = next(iter(formatted_data_dict.values())) if len(formatted_data_dict) == 1 else formatted_data_dict
    elif isinstance(formatted_data_dict, list):
        data_for_df = formatted_data_dict
    else:
        raise ValueError("Formatted data is neither a dictionary nor a list, cannot convert to DataFrame")

    # Create DataFrame
    try:
        df = pd.DataFrame(data_for_df)
        print("DataFrame created successfully.")

        # Save the DataFrame to an Excel file
        excel_output_path = os.path.join(output_folder, f'sorted_data_{timestamp}.xlsx')
        df.to_excel(excel_output_path, index=False)
        print(f"Formatted data saved to Excel at {excel_output_path}")
        
        return df
    except Exception as e:
        print(f"Error creating DataFrame or saving Excel: {str(e)}")
        return None


def calculate_price(prompt_tokens, completion_tokens, model):
    # Define the pricing
    # https://openai.com/api/pricing/

    pricing = {
        "gpt-4o-mini": {
            "input": 0.150 / 1_000_000,  # $0.150 per 1M input tokens
            "output": 0.600 / 1_000_000, # $0.600 per 1M output tokens
        },
        "gpt-4o-2024-08-06": {
            "input": 2.5 / 1_000_000,  # $0.150 per 1M input tokens
            "output": 10 / 1_000_000, # $0.600 per 1M output tokens
        },
        # Add other models and their prices here if needed
    }
    # Calculate the costs
    input_cost = prompt_tokens * pricing[model]["input"]
    output_cost = completion_tokens * pricing[model]["output"]
    total_cost = input_cost + output_cost
    
    return prompt_tokens, completion_tokens, total_cost