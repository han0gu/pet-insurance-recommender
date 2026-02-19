from langchain_core.documents import Document

chunk = Document(
    page_content=('제23조[보험료의 납입이 연체되는 경우 납입최고(독촉)와 계약의 해지]\n'
 '① 계약자가 제2회 이후의 보험료를 납입기일까지 납입하지 않아 보험료 납입이 연체 중인 경우에는, 회사는 14일(보험기간이 1년 미만인 '
 '경우에는 7일) 이상의 기간을 납입최고(독촉)기간으로 정하여 계약자(타인을 위한 계약의 경우 그 특정된 타인을 포함합니다)에게 다음의 '
 '내용을 서면(등기우편 등), 전화(음성녹음) 또는 전자문서 등으로 알려드립니다. 다만, 계약이 해지되기 전에 발생한 보험 금 지급사유에 '
 '대하여 회사는 계약상의 보장을 합니다.'),
    metadata={'source_doc': {'total_pages': 35},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_2_cat_anypet_3_20_21_37.pdf',
         'insurer_code': 'samsung',
         'product_code': '2',
         'product_name': '(일반)반려묘보험 애니펫',
         'total_pages': 35,
         'page': 14},
 'term_type': 'basic',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000065',
              'chunk_char_len': 288,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
