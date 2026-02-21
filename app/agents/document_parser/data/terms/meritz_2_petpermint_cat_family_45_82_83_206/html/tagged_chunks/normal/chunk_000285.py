from langchain_core.documents import Document

chunk = Document(
    page_content=("보험료 관련 용어</h1><br><table id='7' "
 "style='font-size:20px'><thead><tr><td>용어</td><td>정의</td></tr></thead><tbody><tr><td>보험료</td><td>계약자가 "
 '매 납입기일에 납입하기로 한 보험료 로 기본계약 보험료와 특별약관이 부가된 경우 에는 특별약관 보험료의 합계액을 '
 "말합니다.</td></tr></tbody></table><h1 id='8' style='font-size:20px'>제3조(피보험자의 "
 "범위)</h1><br><p id='9'"),
    metadata={'source_doc': {'total_pages': 160},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_2_petpermint_cat_family_45_82_83_206.pdf',
         'insurer_code': 'meritz',
         'product_code': '2',
         'product_name': '무배당 펫퍼민트 Cat&Family보험 다이렉트2601',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'definition', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000285',
              'chunk_char_len': 292,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.55}},
)
