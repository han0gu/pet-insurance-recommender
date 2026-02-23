from langchain_core.documents import Document

chunk = Document(
    page_content=("복리】</h1><br><p id='2' data-category='paragraph' style='font-size:16px'>회사가 "
 '지급할 금전에 이자를 줄 때, 1년마다 마지막 날<br>에 그 이자를 원금에 더한 금액을 다음 1년의 원금으로<br>하는 이자 계산방법을 '
 "말합니다.<br>원금 100원, 이자율 연 10%를 가정할 때</p><br><p id='3' data-category='list' "
 "style='font-size:16px'>- 1년 후 원리금 : 100원 + (100원×10%) = 110원<br>- 2년 후 원리금"),
    metadata={'source_doc': {'total_pages': 160},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_2_petpermint_cat_family_45_82_83_206.pdf',
         'insurer_code': 'meritz',
         'product_code': '2',
         'product_name': '무배당 펫퍼민트 Cat&Family보험 다이렉트2601',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000282',
              'chunk_char_len': 299,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
