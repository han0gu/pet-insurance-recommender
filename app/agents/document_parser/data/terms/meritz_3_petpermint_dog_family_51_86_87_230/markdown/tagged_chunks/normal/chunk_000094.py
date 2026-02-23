from langchain_core.documents import Document

chunk = Document(
    page_content=('에서 정한 사망 당시 계약자적립액(중도인출이 있는 경우\n'
 '중도인출 원금과 이자를 차감하고 적립한 금액을 말합니다)\n'
 '및 미경과보험료를 계약자에게 지급합니다.# 【계약자적립액】장래의 해약환급금 등을 지급하기 위하여 계약자가 납입\n'
 '한 보험료 중 일정액을 기준으로 보험료 및 해약환급금\n'
 '산출방법서에서 정한 방법에 따라 계산한 금액을 말합니\n'
 '다.제5관 보험료의 납입74제26조(제1회 보험료 및 회사의 보장개시)\uf000 회사는 계약의 청약을 승낙하고 제1회 보험료를 받은 '
 '때\n'
 '부터 이 약관이 정한 바에 따라 보장을 합니다. 또한, 회사'),
    metadata={'source_doc': {'total_pages': 180},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_3_petpermint_dog_family_51_86_87_230.pdf',
         'insurer_code': 'meritz',
         'product_code': '3',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000094',
              'chunk_char_len': 290,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.55}},
)
