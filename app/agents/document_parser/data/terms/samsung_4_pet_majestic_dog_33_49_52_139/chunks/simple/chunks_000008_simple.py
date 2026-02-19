from langchain_core.documents import Document

chunk = Document(
    page_content=('<용어풀이>\n'
 '[보험료]\n'
 '보험료는 계약자가 계약에 따라 회사에게 지급하여야 하는 요금을 말하며, 보험료는 「보장보험 료」와 「적립보험료」로 구성되어 있습니다. '
 '또한, 보험료는 보험금 지급을 위한 위험보험료, 회 사가 적립한 금액을 돌려주기 위한 적립부분 순보험료와 부가보험료(회사 운영에 필요한 '
 '계약체결 비용 및 계약관리비용과 보험금 지급조사를 위한 손해조사비)로 구성됩니다.\n'
 '- 보험료 = 보장보험료 + 적립보험료 - 보장보험료 = 위험보험료 + 부가보험료 - 적립보험료 = 적립부분 순보험료 + 부가보험료'),
    metadata={'source_doc': {'total_pages': 107},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_4_pet_majestic_dog_33_49_52_139.pdf',
         'insurer_code': 'samsung',
         'product_code': '4',
         'product_name': '(장기)무배당 삼성화재 펫보험 위풍댕댕',
         'total_pages': 91,
         'page': 33},
 'term_type': 'basic',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000008',
              'chunk_char_len': 283,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
