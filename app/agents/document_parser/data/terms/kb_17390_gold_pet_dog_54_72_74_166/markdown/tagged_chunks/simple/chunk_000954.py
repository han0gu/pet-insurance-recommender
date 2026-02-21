from langchain_core.documents import Document

chunk = Document(
    page_content=('| 주) 1. 만기환급금은 회사가 보험금의 지급시기 도래 7일 이전에 지급할 사유와 금 액을 알리지 않은 경우, 지급사유가 발생한 날의 '
 '다음 날부터 청구일까지의 기간은 공시이율을 적용한 이자를 지급합니다. 2. 지급이자의 계산은 연단위 복리로 계산하며, 금리연동형보험은 '
 '일자 계산합 니다. 단, 보통약관 제1절 일반조항 제45조(소멸시효)에서 정한 소멸시효가 완성된 이후에는 이자를 지급하지 않습니다. 3. '
 '계약자 등의 책임 있는 사유로 보험금 지급이 지연된 때에는 그 해당기간에 대한 이자는 지급되지 않을 수 있습니다'),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000954',
              'chunk_char_len': 290,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
