from langchain_core.documents import Document

chunk = Document(
    page_content=('서 규정한 국내의 병원이나 의원 또는 국외의 의료관련법에\n'
 '서 정한 의료기관에서 발급한 것이어야 합니다.# 제8조(보험금의 지급절차)\uf000 회사는 제7조(보험금의 청구)에서 정한 서류를 '
 '접수한\n'
 '때에는 접수증을 드리고 휴대전화 문자메시지 또는 전자우\n'
 '편 등으로도 송부하며, 그 서류를 접수한 날부터 3영업일\n'
 '이내에 보험금을 지급합니다.\uf000 회사가 보험금 지급사유를 조사ㆍ확인하기 위해 필요한52기간이 제1항의 지급기일을 초과할 것이 '
 '명백히 예상되는\n'
 '경우에는 그 구체적인 사유와 지급예정일 및 보험금 가지급'),
    metadata={'source_doc': {'total_pages': 160},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_2_petpermint_cat_family_45_82_83_206.pdf',
         'insurer_code': 'meritz',
         'product_code': '2',
         'product_name': '무배당 펫퍼민트 Cat&Family보험 다이렉트2601',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'claim', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000023',
              'chunk_char_len': 277,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.55}},
)
