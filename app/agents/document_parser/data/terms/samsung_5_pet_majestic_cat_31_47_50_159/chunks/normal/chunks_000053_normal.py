from langchain_core.documents import Document

chunk = Document(
    page_content=('제14조 (보험수익자의 지정)\n'
 '① 보험수익자를 지정하지 않은 때에는 보험수익자를 제11조(만기환급금의 지급) 제1항 의 경우는 계약자로 하고, 사망보험금의 경우는 '
 '피보험자의 법정상속인, 기타 보험금'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 107,
         'page': 35},
 'term_type': 'basic',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000053',
              'chunk_char_len': 110,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
