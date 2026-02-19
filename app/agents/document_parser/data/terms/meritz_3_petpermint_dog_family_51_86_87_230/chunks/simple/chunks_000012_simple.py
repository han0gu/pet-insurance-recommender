from langchain_core.documents import Document

chunk = Document(
    page_content=('보험료 = 보장보험료 + 적립보험료 보장보험료 = 위험보험료 + 부가보험료 적립보험료 = 적립부분 순보험료 + 부가보험료\n'
 '제2관 보험금의 지급\n'
 '제3조(보험금의 지급사유)\n'
 '회사는 보험증권에 기재된 피보험자가 보험기간 중에 상해 로【별표2(장해분류표)】에서 정한 장해지급률이 80%이상에 해당하는 장해상태가 '
 '되었을 때에는 보험수익자에게 최초 1 회에 한하여 이 보장의 보험가입금액 전액을 일반상해80%이 상후유장해보험금으로 지급합니다.\n'
 '제4조(보험금 지급에 관한 세부규정)'),
    metadata={'source_doc': {'total_pages': 180},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_3_petpermint_dog_family_51_86_87_230.pdf',
         'insurer_code': 'meritz',
         'product_code': '3',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 180,
         'page': 54},
 'term_type': 'basic',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000012',
              'chunk_char_len': 262,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
