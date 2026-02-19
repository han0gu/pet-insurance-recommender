from langchain_core.documents import Document

chunk = Document(
    page_content=('. \uf000 회사는 제1항의 갱신제한 사유 및 제3항의 갱신계약 보 험료에 대하여 갱신전 계약의 보험기간이 끝나기 15일 전까 지 그 '
 '내용을 계약자에게 서면, 전화 또는 전자문서 등으로 안내합니다. \uf000 제3항 및 제4항에도 불구하고 법령 및 표준약관 변경으 로 '
 '보장내용 등이 변경되어 약관이 개정되는 경우 보험기간'),
    metadata={'source_doc': {'total_pages': 180},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_3_petpermint_dog_family_51_86_87_230.pdf',
         'insurer_code': 'meritz',
         'product_code': '3',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 180,
         'page': 183},
 'term_type': 'special',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000623',
              'chunk_char_len': 172,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
