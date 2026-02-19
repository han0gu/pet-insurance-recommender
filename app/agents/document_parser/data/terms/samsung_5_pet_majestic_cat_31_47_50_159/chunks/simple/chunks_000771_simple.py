from langchain_core.documents import Document

chunk = Document(
    page_content=('③ 제1항 제4호의 사고증명서는 수의사법 제12조(진단서 등)에서 규정한 내용에 따라 국 내의 동물병원에서 수의사에 의해 발급한 것이어야 '
 '합니다.\n'
 '<관련법규>\n'
 '[수의사법 제12조(진단서 등)] ① 수의사는 자기가 직접 진료하거나 검안하지 아니하고는 진단서, 검안서, 증명서 또는 처방전( 「'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 107,
         'page': 119},
 'term_type': 'special',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000771',
              'chunk_char_len': 162,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
