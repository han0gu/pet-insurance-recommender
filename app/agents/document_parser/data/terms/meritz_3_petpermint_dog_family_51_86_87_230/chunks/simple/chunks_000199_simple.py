from langchain_core.documents import Document

chunk = Document(
    page_content=('\uf000 회사는 제4조(보험금의 청구)에서 정한 서류를 접수한 때에는 접수증을 드리고 휴대전화 문자메시지 또는 전자우 편 등으로도 '
 '송부하며, 그 서류를 접수한 날부터 3영업일 이내에 보험금을 지급합니다. \uf000 회사가 보험금 지급사유를 조사ㆍ확인하기 위해 필요한 '
 '기간이 제1항의 지급기일을 초과할 것이 명백히 예상되는 경우에는 그 구체적인 사유와 지급예정일 및 보험금 가지급 제도(회사가 추정하는 '
 '보험금의 50% 이내를 지급)에 대하여 피보험자에게 즉시 통지합니다'),
    metadata={'source_doc': {'total_pages': 180},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_3_petpermint_dog_family_51_86_87_230.pdf',
         'insurer_code': 'meritz',
         'product_code': '3',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 180,
         'page': 93},
 'term_type': 'special',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000199',
              'chunk_char_len': 254,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
