from langchain_core.documents import Document

chunk = Document(
    page_content=('- 그 감액된 부분은 해지된 것으로 보며, 이로써 회사가 지급하여야 할 해약환급금이 있\n'
 '- 을 때에는 제35조(해약환급금) 제1항에 따른 해약환급금을 계약자에게 지급합니다.\n'
 '- 다만, 보험가입금액을 감액할 때 해약환급금이 없거나 최초 가입할 때 안내한 해약환\n'
 '- 급금보다 적어질 수 있습니다.\n'
 '- ④ 계약자가 제2항에 따라 보험수익자를 변경하고자 할 경우 계약자와 피보험자가 동일\n'
 '- 하지 않을 때에는 보험금의 지급사유가 발생하기 전에 피보험자가 서면(「전자서명'),
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
 'clause': {'clause_type': 'coverage', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000204',
              'chunk_char_len': 258,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.55}},
)
