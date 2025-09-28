import streamlit as st
from rag_groq import answer

st.set_page_config(page_title="RAG — სამოქალაქო კოდექსი (Matsne)")
st.title("RAG Agent — სამოქალაქო კოდექსი")

q = st.text_area("ჩაწერეთ კითხვა ქართულად", height=140)
if st.button("უპასუხეთ"):
    if not q.strip():
        st.warning("შეიყვანეთ კითხვა.")
    else:
        with st.spinner("ვპასუხობთ..."):
            out = answer(q)
        st.markdown("### პასუხი")
        st.write(out)
