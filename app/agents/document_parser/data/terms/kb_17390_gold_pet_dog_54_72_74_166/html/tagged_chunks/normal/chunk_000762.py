from langchain_core.documents import Document

chunk = Document(
    page_content=('. 그러나 음식물 섭취로 인한 증상, 세균성 음식물 중독과 상습적으로 흡입, 흡수 또는 섭취한 결과로 생긴 중독 증상은 포함되지 '
 '않습니다.</td></tr><tr><td>질병</td><td>상해를 제외한 상병을 모두 포함합니다.</td></tr><tr><td>중요한 '
 '사항</td><td>계약전 알릴 의무와 관련하여 회사가 그 사실을 알았더라면 계약의 청약을 거절하거나 보험가입금액 한도 제한, 일부 보 '
 '장 제외, 보험금 삭감, 보험료 할증과 같이 조건부로 승낙하 는 등 계약 승낙에 영향을 미칠 수 있는 사항을'),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'limit', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000762',
              'chunk_char_len': 285,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
