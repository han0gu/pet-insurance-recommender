from langchain_core.documents import Document

chunk = Document(
    page_content=('- ② 제1항의 경우 피보험자가 동일한 상해의 치료를 직접적인 목적으로 2회 이상 입원한\n'
 '- 경우 이를 1회 입원으로 보아 입원일수를 더합니다.\n'
 '- ③ 제1항의 경우 피보험자가 병원 또는 의원을 이전하여 입원한 경우에도 동일한 상해의\n'
 '- 치료를 직접적인 목적으로 입원한 경우에는 계속하여 입원한 것으로 보아 각 입원일\n'
 '- 수를 더합니다.\n'
 '- ④ 제1항의 경우 피보험자가 보장개시일(책임개시일) 이후 입원하여 치료를 받던 중 보험\n'
 '- 기간이 끝났을 때에도 퇴원하기 전까지의 계속중인 입원에 대하여는 제1항에 따라 상'),
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
 'indexing': {'chunk_id': 'chunk_000378',
              'chunk_char_len': 287,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.55}},
)
