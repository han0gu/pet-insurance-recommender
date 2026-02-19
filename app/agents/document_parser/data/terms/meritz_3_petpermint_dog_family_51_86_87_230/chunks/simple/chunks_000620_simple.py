from langchain_core.documents import Document

chunk = Document(
    page_content=('. \uf000 반려동물 비용손해 관련 특별약관 일반조항 제18조(보험 료의 납입을 연체하여 해지된 계약의 부활(효력회복))에 따 라 이 '
 '계약이 부활이 이루어진 경우에는 부활계약을 제2항 의 최초계약으로 봅니다.(부활(효력회복)이 여러차례 발생 된 경우에는 각각의 '
 '부활(효력회복)계약을 최초계약으로 봅 니다)'),
    metadata={'source_doc': {'total_pages': 180},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_3_petpermint_dog_family_51_86_87_230.pdf',
         'insurer_code': 'meritz',
         'product_code': '3',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 180,
         'page': 183},
 'term_type': 'special',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000620',
              'chunk_char_len': 167,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
