from langchain_core.documents import Document

chunk = Document(
    page_content=('1. 회사가 최초계약 체결당시에 그 사실을 알았거나 과실로 인하여 알지 못하였을 때 2. 회사가 그 사실을 안 날부터 1개월 이상 '
 '지났거나 또는 제1회 보험료를 받은 때부 터 보험금 지급사유가 발생하지 않고 2년(진단계약의 경우 질병에 대하여는 1년) 이 지났을 때 '
 '3. 최초계약을 체결한 날부터 3년이 지났을 때 4'),
    metadata={'source_doc': {'total_pages': 78},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_3_direct_good_pet_28_42_45_105.pdf',
         'insurer_code': 'samsung',
         'product_code': '3',
         'product_name': '(장기)무배당 삼성화재 다이렉트 '
                         '착한펫보험(강아지)',
         'total_pages': 67,
         'page': 50},
 'term_type': 'special',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000200',
              'chunk_char_len': 177,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.55}},
)
