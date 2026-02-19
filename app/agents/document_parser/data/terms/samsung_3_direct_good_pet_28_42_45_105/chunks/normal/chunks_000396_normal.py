from langchain_core.documents import Document

chunk = Document(
    page_content=('③ 제1항에 따라 이 특별약관을 해지하였을 때에는 이 특별약관의 해약환급금을 계약자 에게 지급합니다.\n'
 '④ 제1항 제1호에 의한 이 특별약관의 해지가 보험금 지급사유 발생 후에 이루어진 경우\n'
 '에 회사는 보험금을 지급하지 않으며, 계약 전 알릴 의무 위반 사실(계약해지 등의 원'),
    metadata={'source_doc': {'total_pages': 78},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_3_direct_good_pet_28_42_45_105.pdf',
         'insurer_code': 'samsung',
         'product_code': '3',
         'product_name': '(장기)무배당 삼성화재 다이렉트 '
                         '착한펫보험(강아지)',
         'total_pages': 67,
         'page': 72},
 'term_type': 'special',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000396',
              'chunk_char_len': 153,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
