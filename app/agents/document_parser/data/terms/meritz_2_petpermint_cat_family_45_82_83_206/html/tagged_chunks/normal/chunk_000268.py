from langchain_core.documents import Document

chunk = Document(
    page_content=("id='77' style='font-size:16px'>\uf000 계약관련 용어</h1><br><table id='78' "
 "style='font-size:16px'><thead><tr><td>용어</td><td>정의</td></tr></thead><tbody><tr><td>계약자</td><td>회사와 "
 '계약을 체결하고 보험료를 납입할 의무 를 지는 사람을 말합니다.</td></tr><tr><td>보험 수익자</td><td>보험금 '
 '지급사유가 발생하는 때에 회사에 보험 금을 청구하여 받을 수 있는 사람을'),
    metadata={'source_doc': {'total_pages': 160},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_2_petpermint_cat_family_45_82_83_206.pdf',
         'insurer_code': 'meritz',
         'product_code': '2',
         'product_name': '무배당 펫퍼민트 Cat&Family보험 다이렉트2601',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'definition', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000268',
              'chunk_char_len': 276,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.55}},
)
