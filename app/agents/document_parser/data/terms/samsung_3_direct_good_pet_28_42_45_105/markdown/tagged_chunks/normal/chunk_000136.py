from langchain_core.documents import Document

chunk = Document(
    page_content=('급사유가 발생한 때에는 보험금을 지급하지 않습니다.- 1. 피보험자가 고의로 자신을 해친 경우. 다만, 피보험자가 심신상실 등으로 '
 '자유로운\n'
 '- 의사결정을 할 수 없는 상태에서 자신을 해친 경우에는 보험금의 지급사유에서 정\n'
 '- 한 해당 보험금을 지급합니다.\n'
 '- 2. 보험수익자가 고의로 피보험자를 해친 경우. 다만, 그 보험수익자가 보험금의 일부\n'
 '- 보험수익자인 경우에는 다른 보험수익자에 대한 보험금은 지급합니다.\n'
 '- 3. 계약자가 고의로 피보험자를 해친 경우'),
    metadata={'source_doc': {'total_pages': 78},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_3_direct_good_pet_28_42_45_105.pdf',
         'insurer_code': 'samsung',
         'product_code': '3',
         'product_name': '(장기)무배당 삼성화재 다이렉트 '
                         '착한펫보험(강아지)',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000136',
              'chunk_char_len': 257,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
