from langchain_core.documents import Document

chunk = Document(
    page_content=('서에서 질문한 사항에 대하여 알고 있는 사실을 반드시 사실대로 알려야(이하 「계약 전 알릴 의무」 라 하며, 상법상 「고지의무」 와 '
 '같습니다) 합니다.\n'
 '<관련법규>\n'
 '[상법 제651조(고지의무위반으로 인한 계약해지)]'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 107,
         'page': 101},
 'term_type': 'special',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000576',
              'chunk_char_len': 120,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
