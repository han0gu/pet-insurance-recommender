from langchain_core.documents import Document

chunk = Document(
    page_content=("계약 관련 용어</h1><br><table id='6' "
 "style='font-size:16px'><thead><tr><td>용어</td><td>정의</td></tr></thead><tbody><tr><td>계약자</td><td>회사와 "
 '계약을 체결하고 보험료를 납입할 의 무를 지는 사람을 말합니다.</td></tr><tr><td>기본계약</td><td>계약자와 회사가 '
 '체결한 계약내용 중 보통약 관에 해당하는 부분을 말합니다.</td></tr><tr><td>보험 수익자</td><td>보험금 지급사유가 '
 '발생하는 때에 회사에 보 험금을'),
    metadata={'source_doc': {'total_pages': 160},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_2_petpermint_cat_family_45_82_83_206.pdf',
         'insurer_code': 'meritz',
         'product_code': '2',
         'product_name': '무배당 펫퍼민트 Cat&Family보험 다이렉트2601',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000002',
              'chunk_char_len': 298,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
