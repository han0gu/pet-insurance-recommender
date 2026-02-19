from langchain_core.documents import Document

chunk = Document(
    page_content=('. 다만, 이 계약에서 정한 보험계 약대출금이 있는 때에는 그 원금과 이자의 합계액을 공제한 후의 잔액을 기준으로 합니다. \uf000 '
 '제1항의 중도인출금을 지급받은 경우에는「보험료 및 해 약환급금 산출방법서」에 따라 계약자적립액에서 해당 중도 인출금을 차감합니다.'),
    metadata={'source_doc': {'total_pages': 180},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_3_petpermint_dog_family_51_86_87_230.pdf',
         'insurer_code': 'meritz',
         'product_code': '3',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 180,
         'page': 82},
 'term_type': 'basic',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000157',
              'chunk_char_len': 143,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
