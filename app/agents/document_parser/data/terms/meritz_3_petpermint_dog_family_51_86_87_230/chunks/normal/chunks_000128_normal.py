from langchain_core.documents import Document

chunk = Document(
    page_content=('【자동대출납입】\n'
 '보험료를 제때에 납입하기 곤란한 경우에 계약자가 자동대 출납입을 신청하면 해당 보험 상품의 해약환급금 범위 내 에서 납입할 보험료를 '
 '자동적으로 대출하여 이를 보험료 납입에 충당하는 서비스를 말합니다.\n'
 '제29조(보험료의 납입이 연체되는 경우 납입최고(독촉)와 계약의 해지) \uf000 계약자가 제2회 이후의 보험료를 납입기일까지 납입하지 '
 '않아 보험료 납입이 연체 중인 경우에 회사는 14일(보험기 간이 1년 미만인 경우에는 7일) 이상의 기간을 납입최고(독 '
 '촉)기간(납입최고(독촉)기간의 마지막 날이 영업일이 아닌'),
    metadata={'source_doc': {'total_pages': 180},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_3_petpermint_dog_family_51_86_87_230.pdf',
         'insurer_code': 'meritz',
         'product_code': '3',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 180,
         'page': 76},
 'term_type': 'basic',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000128',
              'chunk_char_len': 291,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
