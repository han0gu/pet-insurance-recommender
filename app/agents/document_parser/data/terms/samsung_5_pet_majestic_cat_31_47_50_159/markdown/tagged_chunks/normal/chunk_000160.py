from langchain_core.documents import Document

chunk = Document(
    page_content=('용합니다.# 제 7조 (보험금을 지급하지 않는 사유)① 회사는 다음 중 어느 한 가지로 각 특별약관별 보험금의 지급사유에서 정한 보험금 '
 '지\n'
 '급사유가 발생한 때에는 보험금을 지급하지 않습니다.- 1. 피보험자가 고의로 자신을 해친 경우. 다만, 피보험자가 심신상실 등으로 '
 '자유로운\n'
 '- 의사결정을 할 수 없는 상태에서 자신을 해친 경우에는 보험금의 지급사유에서 정\n'
 '- 한 해당 보험금을 지급합니다.\n'
 '- 2. 보험수익자가 고의로 피보험자를 해친 경우. 다만, 그 보험수익자가 보험금의 일부'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000160',
              'chunk_char_len': 270,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
