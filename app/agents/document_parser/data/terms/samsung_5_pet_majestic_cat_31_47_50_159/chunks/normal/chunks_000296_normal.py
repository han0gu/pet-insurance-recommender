from langchain_core.documents import Document

chunk = Document(
    page_content=('관 내용의 변경 등) 제1항의 절차에 따라 계약자 명의를 보험수익자로 변경하여 특별 약관의 특별부활(효력회복)을 청약할 수 있음을 '
 '보험수익자에게 통지하여야 합니다.\n'
 '<용어풀이>\n'
 '[강제집행과 담보권실행]'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 107,
         'page': 61},
 'term_type': 'special',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000296',
              'chunk_char_len': 112,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.85}},
)
