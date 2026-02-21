from langchain_core.documents import Document

chunk = Document(
    page_content=('최저보증이율은 연복리 0.3%로 합니<br>다.<br>\uf000 회사는 제1항부터 제3항까지의 규정에서 정한 [보장]공<br>시이율을 '
 '매월 회사의 인터넷 홈페이지 등을 통해 공시합니<br>다.<br>\uf000 회사는 사업연도가 끝나는 날을 기준으로 1년이상 '
 '유지<br>된 계약에 대하여 계약자에게 연1회이상 [보장]공시이율의<br>변경내역을 통지합니다.<br>\uf000 세부적인 '
 '[보장]공시이율의 운영방법은 회사에서 별도로<br>정한「[보장]공시이율 적용에 관한 세부지침」을 따릅니다.</p><br><p '
 "id='72'"),
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
 'indexing': {'chunk_id': 'chunk_000055',
              'chunk_char_len': 277,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.55}},
)
