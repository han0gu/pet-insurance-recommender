from langchain_core.documents import Document

chunk = Document(
    page_content=('- ⑤ 제1항의 경우 피보험자가 병원 또는 의원을 이전하여 입원한 경우에도 동일한 상해의\n'
 '- 치료를 직접 목적으로 입원한 경우에는 계속하여 입원한 것으로 보아 각 입원일수를\n'
 '- 더합니다.\n'
 '- ⑥ 제1항의 경우 피보험자가 보장개시일(책임개시일) 이후 입원하여 치료를 받던 중 보험\n'
 '- 기간이 끝났을 때에도 퇴원하기 전까지의 계속중인 입원에 대하여는 제1항에 따라 반\n'
 '- 려견 위탁비용을 계속 보장합니다.\n'
 '- ⑦ 피보험자가 정당한 이유없이 입원기간 중 의사의 지시를 따르지 않은 때에는 회사는'),
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
 'clause': {'clause_type': 'coverage', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000509',
              'chunk_char_len': 273,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.55}},
)
