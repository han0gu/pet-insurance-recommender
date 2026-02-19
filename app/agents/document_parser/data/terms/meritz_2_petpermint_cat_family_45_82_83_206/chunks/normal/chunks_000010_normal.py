from langchain_core.documents import Document

chunk = Document(
    page_content=('\uf000 보험료 관련 용어\n'
 '용어 | 정의\n'
 '보험료 | 계약자가 매 납입기일에 납입하기로 한 보험 료로 기본계약 보장보험료 및 적립보험료와 특별약관이 부가된 경우에는 특별약관 보험료 '
 '의 합계액을 말합니다.\n'
 '보장 보험료 | 계약에서 정한 보험금을 지급하는데 필요한 보험료를 말합니다.\n'
 '적립 보험료 | 회사가 적립한 금액을 돌려주는데 필요한 보 험료를 말합니다.\n'
 '【보험료】'),
    metadata={'source_doc': {'total_pages': 160},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_2_petpermint_cat_family_45_82_83_206.pdf',
         'insurer_code': 'meritz',
         'product_code': '2',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 160,
         'page': 49},
 'term_type': 'basic',
 'clause': {'clause_type': 'definition', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000010',
              'chunk_char_len': 201,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.55}},
)
