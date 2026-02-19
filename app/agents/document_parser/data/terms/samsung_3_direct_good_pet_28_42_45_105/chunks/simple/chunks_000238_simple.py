from langchain_core.documents import Document

chunk = Document(
    page_content=('③ 회사는 계약자가 제1항에 따라 이 특별약관의 보험가입금액을 감액하고자 할 때에는 그 감액된 부분은 해지된 것으로 보며, 이로써 회사가 '
 '지급하여야 할 해약환급금이 있 을 때에는 제35조(해약환급금) 제1항에 따른 해약환급금을 계약자에게 지급합니다. 다만, 보험가입금액을 '
 '감액할 때 해약환급금이 없거나 최초 가입할 때 안내한 해약환 급금보다 적어질 수 있습니다'),
    metadata={'source_doc': {'total_pages': 78},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_3_direct_good_pet_28_42_45_105.pdf',
         'insurer_code': 'samsung',
         'product_code': '3',
         'product_name': '(장기)무배당 삼성화재 다이렉트 '
                         '착한펫보험(강아지)',
         'total_pages': 66,
         'page': 53},
 'term_type': 'special',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000238',
              'chunk_char_len': 201,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
