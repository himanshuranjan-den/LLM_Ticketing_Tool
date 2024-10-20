import streamlit as st

tab_titles = ['HR Support', 'IT Suport', 'Transportation Support']
tabs = st.tabs(tabs=tab_titles)

with tabs[0]:
    st.title("HR Support Tickets")
    for tickets in st.session_state["HR_tickets"]:
        st.write(str(st.session_state["HR_tickets"].index(tickets)+1) + " : " + tickets)

with tabs[1]:
    st.title("IT Support Tickets")
    for tickets in st.session_state["IT_tickets"]:
        st.write(str(st.session_state["IT_tickets"].index(tickets)+1) + " : " + tickets)

with tabs[2]:
    st.title("Transportation Support Tickets")
    for tickets in st.session_state["Transport_tickets"]:
        st.write(str(st.session_state["Transport_tickets"].index(tickets)+1) + " : " + tickets)