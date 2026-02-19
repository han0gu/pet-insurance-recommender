from langchain_core.documents import Document

chunk = Document(
    page_content=('부담하지 않습니다.\n'
 '㉤ 피보험자가「배상책임 관련 특별약관 일반조항」제9 조(손해배상청구에 대한 회사의 해결)의 제2항 및 제3항의 회사의 요구에 따르기 '
 '위하여 지출한 비용\n'
 '\uf000 제4항의 손해에 대하여 다음과 같이 보상합니다.'),
    metadata={'source_doc': {'total_pages': 180},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_3_petpermint_dog_family_51_86_87_230.pdf',
         'insurer_code': 'meritz',
         'product_code': '3',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 180,
         'page': 187},
 'term_type': 'special',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000635',
              'chunk_char_len': 124,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
