from langchain_core.documents import Document

chunk = Document(
    page_content=('- 내일 것\n'
 '- 3. 보통약관 제30조(보험료의 납입이 연체되는 경우 납입최고(독촉)와 계약의 해지)에\n'
 '- 정한 납입최고(독촉)기간 내에 갱신전 계약의 보험료가 납입완료 되었을 것\n'
 '- ② 제1항에 따라 정상적으로 갱신이 이루어진 경우 갱신계약의 보장은 갱신전 계약에 의\n'
 '- 한 보장이 끝나는 때부터 적용합니다.\n'
 '- ③ 제1항에도 불구하고 갱신전 계약에서 소멸사유가 발생한 경우에는 해당 갱신형 계약\n'
 '- 은 갱신되지 않습니다.\n'
 '- ④ 제3항에도 불구하고 보험금 청구 지연 등의 사유로 갱신이 이루어진 경우에는 해당'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000657',
              'chunk_char_len': 286,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
