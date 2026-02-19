from langchain_core.documents import Document

chunk = Document(
    page_content=('1. 피보험자가 고의로 자신을 해친 경우. 다만, 피보험자가 심신상실 등으로 자유로운 의사결정을 할 수 없는 상태에서 자신을 해친 '
 '경우에는 보험금의 지급사유에서 정 한 해당 보험금을 지급합니다. 2. 보험수익자가 고의로 피보험자를 해친 경우. 다만, 그 보험수익자가 '
 '보험금의 일부 보험수익자인 경우에는 다른 보험수익자에 대한 보험금은 지급합니다. 3. 계약자가 고의로 피보험자를 해친 경우 4. 전쟁, '
 '외국의 무력행사, 혁명, 내란, 사변, 폭동'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 107,
         'page': 71},
 'term_type': 'special',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000373',
              'chunk_char_len': 249,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
