from langchain_core.documents import Document

chunk = Document(
    page_content=('\uf000 갱신보장계약의 보험기간은 갱신전 보장계약의 보험기간 과 동일한 것으로 합니다. 다만, 갱신일의 피보험자의 보험 나이부터 '
 '갱신종료보험나이(갱신시점의 갱신종료보험나이를 말합니다)까지의 기간이 갱신전 보장계약의 보험기간 미만 인 경우에는 그 잔여기간을 '
 '보험기간으로 합니다.\n'
 '제3조(자동갱신 적용)\n'
 '\uf000 회사는 갱신계약에 대하여 갱신전 약관을 적용하며 보험 요율에 관한 제도 또는 보험료 등을 개정한 경우에는 갱신 보장계약에 '
 '대해서는 갱신일 현재의 제도 또는 보험료 등을 적용합니다.'),
    metadata={'source_doc': {'total_pages': 180},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_3_petpermint_dog_family_51_86_87_230.pdf',
         'insurer_code': 'meritz',
         'product_code': '3',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 180,
         'page': 189},
 'term_type': 'special',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000647',
              'chunk_char_len': 267,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
