from langchain_core.documents import Document

chunk = Document(
    page_content=('【계약자적립액】\n'
 '장래의 해약환급금 등을 지급하기 위하여 계약자가 납입 한 보험료 중 일정액을 기준으로 보험료 및 해약환급금 산출방법서에서 정한 방법에 '
 '따라 계산한 금액을 말합니 다.\n'
 '제5관 보험료의 납입'),
    metadata={'source_doc': {'total_pages': 180},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_3_petpermint_dog_family_51_86_87_230.pdf',
         'insurer_code': 'meritz',
         'product_code': '3',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 180,
         'page': 74},
 'term_type': 'basic',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000116',
              'chunk_char_len': 113,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
