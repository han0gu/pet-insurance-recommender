from langchain_core.documents import Document

chunk = Document(
    page_content=('를 납입한 경우에는 영수증을 발행하여 드립니다. 다만, 금융회사(우체국을 포함합니다)를\n'
 '통하여 보험료를 납입한 경우에는 그 금융회사 발행 증빙서류를 영수증으로 대신합니다.【납입기일】계약자가 제2회 이후의 보험료를 납입하기로 '
 '한 날을 말합니다.- 15 -제27조(보험료의 납입이 연체되는 경우 납입최고(독촉)와 계약의 해지)① 계약자가 제2회 이후의 보험료를 '
 '납입기일까지 납입하지 않아 보험료 납입이 연체 중\n'
 '인 경우에 회사는 14일(보험기간이 1년 미만인 경우에는 7일) 이상의 기간을 납입최고'),
    metadata={'source_doc': {'total_pages': 50},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_1_maum_pet_1_21_22_50.pdf',
         'insurer_code': 'meritz',
         'product_code': '1',
         'product_name': '메리츠 마음든든 반려동물보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000084',
              'chunk_char_len': 276,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
