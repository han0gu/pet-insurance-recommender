from langchain_core.documents import Document

chunk = Document(
    page_content=('\uf000 제1항 제2호의 사고증명서는 의료법 제3조(의료기관)에 서 규정한 국내의 병원이나 의원 또는 국외의 의료관련법에 서 정한 '
 '의료기관에서 발급한 것이어야 합니다.\n'
 '제8조(보험금의 지급절차)\n'
 '\uf000 회사는 제7조(보험금의 청구)에서 정한 서류를 접수한 때에는 접수증을 드리고 휴대전화 문자메시지 또는 전자우 편 등으로도 '
 '송부하며, 그 서류를 접수한 날부터 3영업일 이내에 보험금을 지급합니다.\n'
 '\uf000 회사가 보험금 지급사유를 조사ㆍ확인하기 위해 필요한'),
    metadata={'source_doc': {'total_pages': 160},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_2_petpermint_cat_family_45_82_83_206.pdf',
         'insurer_code': 'meritz',
         'product_code': '2',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 160,
         'page': 52},
 'term_type': 'basic',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000026',
              'chunk_char_len': 247,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
