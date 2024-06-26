import streamlit as st
import brx
import spacy
from spacy_streamlit import load_model
import asyncio

st.title("Automatic Entity Extraction and Classification")
entity_types = st.multiselect("What kinds of words/phrases do you want to select?", [
    "PERSON - People, including fictional",
    "NORP - Nationalities/religious groups/political groups",
    "FAC - Buildings, airports, highways, bridges, etc.",
    "ORG - Companies, agencies, institutions, etc.",
    "GPE - Countries, cities, etc.",
    "LOC - Non-GPE locations, mountain ranges, bodies of water",
    "PRODUCT - Objects, vehicles, foods, etc. (Not services.)",
    "EVENT - Named hurricanes, battles, wars, sports events, etc.",
    "LAW - Named documents made into laws",
    "LANGUAGE - Any named language",
    "DATE - Absolute or relative dates or periods",
    "TIME - Times smaller than a day",
    "PERCENT - Percentage, including ”%“",
    "MONEY - Monetary values, including unit",
    "QUANTITY - Measurements, as of weight or distance",
    "ORDINAL - “first”, “second”, etc.",
    "CARDINAL - Numerals that do not fall under another type",
])

extraction_rules = st.text_input("State how each entity should be classified.")

text = st.text_area("Enter text to process.")

import json
import os

brx_client = brx.BRX(
    access_token=os.environ.get("BRX_ACCESS_TOKEN"),
    verbose=False
)

def apply_dict_to_if(input_dict, input_fields):
    """
    Applies a given dictionary to input_fields so that you can directly send a dictionary to BRX
    """
    for index, field in enumerate(input_fields):
        for dict_key in input_dict.keys():
            if field["name"] == dict_key:
                input_fields[index]["value"] = input_dict[dict_key]
    return input_fields

def call_brk(data):
    with st.spinner("Generating response..."):
        print("Test")
        result = brx_client.run_sfid_with_dict(
            "b75bc574-ce23-42ed-a5de-105aa1b4b72d",
            data
        )
        result = json.loads(result[0])
        print(result)
        try:
            result = result["brxRes"]["output"]
            st.success(f"{data['text']} - {result}\n")
            return result
        except Exception as e:
            print(e)
            print("Result: ", result)
            st.write(f"Error result: {str(result)}")
            return None

if st.button("Process"):
    # check for length
    if len(text) > 10000:
        st.write("Your text is too long!")
    else:
        # extract relevant items from text
        nlp = load_model("en_core_web_sm")
        doc = nlp(text)
        to_classify = []
        tasks = []
        print("Separate entities: ", doc.ents)
        for entity in doc.ents:
            for entity_type in entity_types:
                if entity.label_ == entity_type.split("-")[0].strip():
                    to_classify.append(entity.text)
                    call_brk( 
                        {"classification_rules": extraction_rules, "text": entity.text}
                    )