from langchain_core.documents import Document

chunk = Document(
    page_content=('대위권 | 회사가 보험금을 지급하고 취득하는 법률상의 권리를 말합니다.\n'
 '\uf000 지급금과 이자율 관련 용어\n'
 '용어 | 정의\n'
 '연단위 복리 | 회사가 지급할 금전에 이자를 줄 때 1년마다 마지막 날에 그 이자를 원금에 더한 금액을 다 음 1년의 원금으로 하는 이자 '
 '계산방법을 말합 니다.\n'
 '계약자 적립액 | 장래의 해약환급금 등을 지급하기 위하여 계약 자가 납입한 보험료 중 일정액을 기준으로 보 험료 및 해약환급금 '
 '산출방법서에서 정한 방법 에 따라 계산한 금액을 말합니다.'),
    metadata={'source_doc': {'total_pages': 180},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_3_petpermint_dog_family_51_86_87_230.pdf',
         'insurer_code': 'meritz',
         'product_code': '3',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 180,
         'page': 175},
 'term_type': 'special',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000584',
              'chunk_char_len': 258,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
