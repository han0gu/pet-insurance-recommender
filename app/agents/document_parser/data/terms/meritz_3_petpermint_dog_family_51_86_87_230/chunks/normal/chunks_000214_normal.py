from langchain_core.documents import Document

chunk = Document(
    page_content=('【위험변경시 해약환급금 정산】\n'
 '제1항에 따라 위험이 증가ㆍ감소되는 경우 이후 기간 보 장을 위한 재원인 계약자적립액 등의 차이로 계약자가 추가로 납입하여야 할(또는 '
 '반환받을) 금액이 발생할 수 있습니다.\n'
 '【계약자적립액】\n'
 '장래의 해약환급금 등을 지급하기 위하여 계약자가 납입 한 보험료 중 일정액을 기준으로 보험료 및 해약환급금 산출방법서에서 정한 방법에 '
 '따라 계산한 금액을 말합니 다.'),
    metadata={'source_doc': {'total_pages': 180},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_3_petpermint_dog_family_51_86_87_230.pdf',
         'insurer_code': 'meritz',
         'product_code': '3',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 180,
         'page': 96},
 'term_type': 'special',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000214',
              'chunk_char_len': 215,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
