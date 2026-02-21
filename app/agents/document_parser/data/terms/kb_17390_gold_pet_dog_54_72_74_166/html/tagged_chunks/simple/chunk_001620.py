from langchain_core.documents import Document

chunk = Document(
    page_content=('합산하여 정상운동영역의 1/2 이하이거나 중수지관절의 굴<br>신(굽히고 펴기)운동영역이 정상운동영역의 1/2 이하인 경우를 '
 '말한다.<br>8) 한 손가락에 장해가 생기고 다른 손가락에 장해가 발생한 경우, 지급률<br>은 각각 적용하여 합산한다.<br>9) '
 '손가락의 관절기능장해 평가는 손가락 관절의 관절운동범위 제한 등으로<br>평가한다'),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['digestive', 'joint']},
 'indexing': {'chunk_id': 'chunk_001620',
              'chunk_char_len': 190,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
