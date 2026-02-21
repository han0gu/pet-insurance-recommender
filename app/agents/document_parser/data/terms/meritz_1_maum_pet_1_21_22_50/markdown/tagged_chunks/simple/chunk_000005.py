from langchain_core.documents import Document

chunk = Document(
    page_content=('- 해를 말합니다. 유독가스 또는 유독물질을 우연히 일시적으로 흡입, 흡수 또는 섭\n'
 '- 취한 결과로 발생하는 중독 증상을 포함합니다. 그러나 세균성 음식물 중독과 상습\n'
 '- 적으로 흡입, 흡수 또는 섭취한 결과로 생긴 중독증상은 이에 포함되지 않습니다.\n'
 '- 나. 질병: 상해를 제외한 상병을 모두 포함합니다.\n'
 '- 다. 중요한 사항: 계약전 알릴 의무와 관련하여 회사가 그 사실을 알았더라면 계약의\n'
 '- 청약을 거절하거나 보험가입금액 한도 제한, 일부 보장 제외, 보험금 삭감, 보험료'),
    metadata={'source_doc': {'total_pages': 50},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_1_maum_pet_1_21_22_50.pdf',
         'insurer_code': 'meritz',
         'product_code': '1',
         'product_name': '메리츠 마음든든 반려동물보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000005',
              'chunk_char_len': 268,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
