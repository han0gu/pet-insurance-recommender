from langchain_core.documents import Document

chunk = Document(
    page_content=('상위목뼈(상위경추: 제1, 2경추) CT 검사상, 환 추 전방 궁(arch)의 후방과 치상돌기의 전면과의 거리(ADI: '
 'Atlanto-Dental Interval)에 뚜렷한 이상전위가 있는 상태'),
    metadata={'source_doc': {'total_pages': 180},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_3_petpermint_dog_family_51_86_87_230.pdf',
         'insurer_code': 'meritz',
         'product_code': '3',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 180,
         'page': 212},
 'term_type': 'special',
 'clause': {'clause_type': 'definition', 'risk_domains': ['joint']},
 'indexing': {'chunk_id': 'chunk_000751',
              'chunk_char_len': 108,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
