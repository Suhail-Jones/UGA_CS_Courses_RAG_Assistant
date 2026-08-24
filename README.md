# UGA_CS_Courses_RAG_Assistant

A retrieval-augmented generation tool that answers questions about UGA's CS degree requirements and course prerequisites, grounded in real UGA source material.

## Status: In-Progress

- [x] `scrape_courses.py` — scrapes the CSCI course index, filters to undergrad-relevant (1000–4999) courses, saves URLs to `sources.txt`
- [x] `build_index.py` — loads course pages + degree requirement pages, chunks and embeds them locally, builds and persists a FAISS vector store
- [ ] `agent.py` — not started; will load the saved vector store and answer questions via Gemini
