from langchain_core.documents import Document

chunk = Document(
    page_content=('1년 후 원리금 : 100원 + (100원×10%) = 110원<br>- 2년 후 원리금 : 110원 + (110원×10%) = '
 "121원</p><br><h1 id='4' style='font-size:20px'>\uf000 기간과 날짜 관련 "
 "용어</h1><br><table id='5' "
 "style='font-size:16px'><thead><tr><td>용어</td><td>정의</td></tr></thead><tbody><tr><td>보험기간</td><td>계약에 "
 '따라 보장을 받는 기간을'),
    metadata={'source_doc': {'total_pages': 160},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_2_petpermint_cat_family_45_82_83_206.pdf',
         'insurer_code': 'meritz',
         'product_code': '2',
         'product_name': '무배당 펫퍼민트 Cat&Family보험 다이렉트2601',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000283',
              'chunk_char_len': 269,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
