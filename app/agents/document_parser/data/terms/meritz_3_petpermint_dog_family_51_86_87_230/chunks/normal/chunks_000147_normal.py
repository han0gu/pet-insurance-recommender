from langchain_core.documents import Document

chunk = Document(
    page_content=('. \uf000 제1항 및 제3항에 따라 계약이 해지된 경우 회사는 제35 조(해약환급금) 제4항에 따른 해약환급금을 계약자에게 지 '
 '급합니다. \uf000 계약자는 제1항의 제척기간에도 불구하고 민법 등 관계 법령에서 정하는 바에 따라 법률상의 권리를 행사 할 수 있 '
 '습니다.'),
    metadata={'source_doc': {'total_pages': 180},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_3_petpermint_dog_family_51_86_87_230.pdf',
         'insurer_code': 'meritz',
         'product_code': '3',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 180,
         'page': 80},
 'term_type': 'basic',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000147',
              'chunk_char_len': 144,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.9}},
)
