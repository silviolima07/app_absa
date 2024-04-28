import streamlit as st
import os


# App title
st.set_page_config(page_title="🦙 ABSA")


import sentencepiece

from transformers import T5Tokenizer, T5ForConditionalGeneration, pipeline

import accelerate

from datasets import load_dataset

from PIL import Image

#@st.cache_data
#def get_ds():
#    data_files = {'train': "train.csv"}
#    ds = load_dataset('SilvioLima/absa', data_files=data_files)
#    return ds


def get_data(df,column):
    l_data = df[column]
    return l_data

html_page_title = """
     <div style="background-color:tomato;padding=50px">
         <p style='text-align:center;font-size:50px;font-weight:bold'>ABSA</p>
     </div>
               """
st.markdown(html_page_title, unsafe_allow_html=True)

html_page_title2 = """
     <div style="background-color:tomato;padding=50px">
         <p style='text-align:center;font-size:50px;font-weight:bold'>Aspect / Opinion / Polarity</p>
     </div>
               """
st.markdown(html_page_title2, unsafe_allow_html=True)


hf_model = "SilvioLima/absa_model_v1"

hf_ds = "SilvioLima/absa"

data_files = {'train': "train.csv"}
ds = load_dataset('SilvioLima/absa')

df = ds['train'].to_pandas()
skip_special_tokens=True

#st.dataframe(df)

activities = ["Test","About"]
choice = st.sidebar.radio("Test",activities)

pipeline = pipeline(task="text2text-generation", model="SilvioLima/absa_model")

#review = "I dislike this restaurant because foodddd is horrible and price is high."

finetuned_model = T5ForConditionalGeneration.from_pretrained(hf_model)
tokenizer = T5Tokenizer.from_pretrained(hf_model)

instruction = "Identify aspects and opinions in this sentence. Example:\
 - sentence: 'food is great because food is great, environment is great because environment is even better'\
 - Aspect, opinion and polarity:\
 [('food', 'great', 'POS'), ('environment', 'better', 'POS')]"
 
 

if choice == 'Test':
        
    st.sidebar.markdown("### Choose a domain")
    option_domain = st.sidebar.selectbox('Domain',set(get_data(df, 'domain')), label_visibility = 'hidden')
    df_domain = df.loc[df['domain'] == option_domain]
    review = st.selectbox('Sentence',get_data(df, 'sentence'), label_visibility = 'hidden')
    st.subheader(review)
    if st.button("Extract Triples"):
        try:
            with st.spinner('Wait for it...we are processing'):                  
                #generated = pipeline(review)
                #answer = generated[0]
                input_text = instruction + " " + review
                print(input_text)
                # Tokenizar a entrada
                inputs = tokenizer(input_text, max_length=max_length, truncation=True , return_tensors="pt", padding="max_length", skip_special_tokens=True)
                print("Tokenized:", inputs)
                outputs = finetuned_model.generate(**inputs, max_length=max_length) #, num_return_sequences=1, num_beams=5, no_repeat_ngram_size=2, top_k=50, top_p=0.95, temperature=0.0, early_stopping=True
                print("Outputs:",outputs)
                generated_part = tokenizer.decode(outputs[0], skip_special_tokens=True)
                #print("Original generated:", generated_part)
                generated_part = generated_part.replace("s>", "")
                generated_parts.append(generated_part)
                # Juntar as partes geradas em uma única sequência
                generated_sentence = ", ".join(generated_parts)
                st.header(generated_sentence)
        except:
               st.error('Sorry. Try again or wait a moment', icon="🚨")

else:
    
    image2 = Image.open("imgs/project.png")
    #image1 = Image.open("imgs/logo1.png")
    #image3 = Image.open("imgs/logo2.png")
    col1, col2, col3, col4 = st.columns(4)
    
    #with col1:
    #    st.image(image3,caption="", use_column_width=False)
        
    with col2:
        st.image(image2,caption="", use_column_width=False)

    #with col3:
    #    st.image(image1,caption="", use_column_width=False)

    #with col4:
    #    st.image(image2,caption="", use_column_width=False)        






