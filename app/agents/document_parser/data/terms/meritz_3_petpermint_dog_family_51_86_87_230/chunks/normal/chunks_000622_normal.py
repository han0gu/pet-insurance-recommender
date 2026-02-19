from langchain_core.documents import Document

chunk = Document(
    page_content=('\uf000 갱신계약의 보험기간은 갱신전 계약의 보험기간과 동일 한 것으로 합니다. 다만, 갱신일의 반려동물의 만나이로부 터 '
 '갱신종료만나이(갱신시점의 갱신종료만나이를 말합니다) 까지의 기간이 갱신전 계약의 보험기간 미만인 경우에는 그 잔여기간을 보험기간으로 '
 '합니다. \uf000 회사는 갱신계약에 대하여 갱신전 약관을 적용하며, 보 험요율에 관한 제도 또는 보험료 등을 개정한 경우에는 갱 '
 '신계약에 대해서는 갱신일 현재의 제도 또는 보험료 등을 적용합니다'),
    metadata={'source_doc': {'total_pages': 180},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_3_petpermint_dog_family_51_86_87_230.pdf',
         'insurer_code': 'meritz',
         'product_code': '3',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 180,
         'page': 183},
 'term_type': 'special',
 'clause': {'clause_type': 'renewal', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000622',
              'chunk_char_len': 244,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.55}},
)
