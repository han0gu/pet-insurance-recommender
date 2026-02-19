from langchain_core.documents import Document

chunk = Document(
    page_content=('\uf000 회사는 제2조(자동갱신 적용대상 계약의 자동갱신)에서 정한 갱신제한 사유 및 제1항의 갱신보장계약 보험료에 대 하여 갱신대상 '
 '보장계약의 보험기간이 끝나기 15일 전까지 그 내용을 계약자에게 서면, 전화 또는 전자문서 등으로 안 내하여 드립니다.'),
    metadata={'source_doc': {'total_pages': 180},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_3_petpermint_dog_family_51_86_87_230.pdf',
         'insurer_code': 'meritz',
         'product_code': '3',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 180,
         'page': 190},
 'term_type': 'special',
 'clause': {'clause_type': 'renewal', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000648',
              'chunk_char_len': 138,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.55}},
)
