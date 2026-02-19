from langchain_core.documents import Document

chunk = Document(
    page_content=('① 제1항 제1호의 경우에는 그 노력을 하였더라면 손해를 방지 또는 경감할 수 있었던 금액 ② 제1항 제2호의 경우에는 제3자로부터 '
 '손해의 배상을 받을 수 있었던 금액 ③ 제1항 제3호의 경우에는 소송비용(중재 또는 조정에 관한 비용 포함) 및 변호사비용과 회사의 '
 '동의를 받지 않은 행위로 증가된 손해\n'
 '제9조(손해배상청구에 대한 회사의 해결)'),
    metadata={'source_doc': {'total_pages': 180},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_3_petpermint_dog_family_51_86_87_230.pdf',
         'insurer_code': 'meritz',
         'product_code': '3',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 180,
         'page': 179},
 'term_type': 'special',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000600',
              'chunk_char_len': 191,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
