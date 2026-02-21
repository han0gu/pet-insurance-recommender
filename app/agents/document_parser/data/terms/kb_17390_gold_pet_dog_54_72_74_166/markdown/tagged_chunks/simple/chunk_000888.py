from langchain_core.documents import Document

chunk = Document(
    page_content=('나) 육안으로 변형(결손을 포함)을 명백하게 알 수 있을 정도로 방사선\n'
 '검사로 측정한 각(角) 변형이 20° 이상인 경우\n'
 '다) 미골의 기형은 골절이나 탈구로 방사선 검사로 측정한 각(角) 변형\n'
 '이 70° 이상 남은 상태- \n'
 '- 3) ‘빗장뼈(쇄골), 가슴뼈(흉골), 갈비뼈(늑골), 어깨뼈(견갑골)에 뚜렷한\n'
 '- 기형이 남은 때’라 함은 방사선 검사로 측정한 각(角) 변형이 20° 이\n'
 '- 상인 경우를 말한다.\n'
 '- 4) 갈비뼈(늑골)의 기형은 그 개수와 정도, 부위 등에 관계없이 전체를 일괄'),
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
 'indexing': {'chunk_id': 'chunk_000888',
              'chunk_char_len': 274,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
