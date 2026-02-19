from langchain_core.documents import Document

chunk = Document(
    page_content=('② 회사는 제1항의 통지로 인하여 위험의 변동이 발생한 경우에는 보통약관 제24조(계약 내용의 변경 등)에 따라 계약내용을 변경할 수 '
 '있습니다.\n'
 '<유의사항>\n'
 '[위험변경에 따른 계약변경 절차]'),
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
 'indexing': {'chunk_id': 'chunk_000579',
              'chunk_char_len': 106,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
