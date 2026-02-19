from langchain_core.documents import Document

chunk = Document(
    page_content=('\uf000 제1항에도 불구하고 지정된 동물병원에서 진료를 받고 「동물병원 보험금 자동청구」절차를 이용한 경우에는 제1 항의 서류를 '
 '제출한 것으로 간주합니다. 다만, 회사가 보험 금 지급을 위해 필요하다고 인정하는 경우 관련 서류를 요 청할 수 있습니다.'),
    metadata={'source_doc': {'total_pages': 180},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_3_petpermint_dog_family_51_86_87_230.pdf',
         'insurer_code': 'meritz',
         'product_code': '3',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 180,
         'page': 92},
 'term_type': 'special',
 'clause': {'clause_type': 'claim', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000197',
              'chunk_char_len': 137,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
