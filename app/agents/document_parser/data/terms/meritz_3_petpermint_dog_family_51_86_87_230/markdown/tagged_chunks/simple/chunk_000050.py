from langchain_core.documents import Document

chunk = Document(
    page_content=('없음) 동안의 분납 중 선택하여 정산금액을 납입하여야 합\n'
 '니다. 다만, 보험료 갱신형 계약 등 일부 보험계약의 경우\n'
 '분납이 제한될 수 있습니다.# 【위험변경시 해약환급금 정산】제1항에 따라 위험이 증가ㆍ감소되는 경우 이후 기간 보\n'
 '장을 위한 재원인 계약자적립액 등의 차이로 계약자가 추\n'
 '가로 납입하여야 할(또는 반환받을) 금액이 발생할 수 있\n'
 '습니다.# 【계약자적립액】장래의 해약환급금 등을 지급하기 위하여 계약자가 납입\n'
 '한 보험료 중 일정액을 기준으로 보험료 및 해약환급금\n'
 '산출방법서에서 정한 방법에 따라 계산한 금액을 말합니'),
    metadata={'source_doc': {'total_pages': 180},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_3_petpermint_dog_family_51_86_87_230.pdf',
         'insurer_code': 'meritz',
         'product_code': '3',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000050',
              'chunk_char_len': 295,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
