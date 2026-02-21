from langchain_core.documents import Document

chunk = Document(
    page_content=('| 1. 부재자의 생사가 5년간 분명하지 않은 때에는 법원은 이해관계인이나 검사의 별 청구에 의하여 실종선고를 하여야 합니다. 표 2. '
 '전지에 임한 자, 침몰한 선박 중에 있던 자, 추락한 항공기 중에 있던 자, 기 타 사망의 원인이 될 위난을 당한 자의 생사가 전쟁 종지 '
 '후 또는 선박의 침 몰, 항공기의 추락, 기타 위난이 종료한 후 1년간 분명하지 않은 때에도 제1항 과 같습니다. | 1. 부재자의 '
 '생사가 5년간 분명하지 않은 때에는 법원은 이해관계인이나 검사의 별 청구에 의하여 실종선고를 하여야 합니다. 표 2'),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000143',
              'chunk_char_len': 291,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
