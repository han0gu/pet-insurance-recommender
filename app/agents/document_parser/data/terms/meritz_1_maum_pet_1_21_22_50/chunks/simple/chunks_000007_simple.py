from langchain_core.documents import Document

chunk = Document(
    page_content=('. 나. 질병: 상해를 제외한 상병을 모두 포함합니다. 다. 중요한 사항: 계약전 알릴 의무와 관련하여 회사가 그 사실을 알았더라면 '
 '계약의 청약을 거절하거나 보험가입금액 한도 제한, 일부 보장 제외, 보험금 삭감, 보험료 할증과 같이 조건부로 승낙하는 등 계약 승낙에 '
 '영향을 미칠 수 있는 사항을 말합 니다.'),
    metadata={'source_doc': {'total_pages': 50},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_1_maum_pet_1_21_22_50.pdf',
         'insurer_code': 'meritz',
         'product_code': '1',
         'product_name': '메리츠 마음든든 반려동물보험',
         'total_pages': 50,
         'page': 2},
 'term_type': 'basic',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000007',
              'chunk_char_len': 172,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
