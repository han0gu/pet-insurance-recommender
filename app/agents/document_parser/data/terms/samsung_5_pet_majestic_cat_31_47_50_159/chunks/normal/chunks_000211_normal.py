from langchain_core.documents import Document

chunk = Document(
    page_content=('제14조 (대표자의 지정)\n'
 '① 계약자 또는 보험수익자가 2명 이상인 경우에는 각 대표자를 1명 지정하여야 합니다. 이 경우 그 대표자는 각각 다른 계약자 또는 '
 '보험수익자를 대리하는 것으로 합니다. ② 지정된 계약자 또는 보험수익자의 소재가 확실하지 않은 경우에는 이 특별약관에 관 하여 회사가 '
 '계약자 또는 보험수익자 1명에 대하여 한 행위는 각각 다른 계약자 또는 보험수익자에게도 효력이 미칩니다. ③ 계약자가 2명 이상인 '
 '경우에는 그 책임을 연대로 합니다.\n'
 '<예시안내>\n'
 '[계약자가 2명 이상인 경우]'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 107,
         'page': 53},
 'term_type': 'special',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000211',
              'chunk_char_len': 280,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
