from langchain_core.documents import Document

chunk = Document(
    page_content=('- 관절 포함) 심장에서 가까운 쪽으로 손가락이 절단되었을 때를 말한다.\n'
 '- 6) ‘손가락뼈 일부를 잃었을 때’라 함은 첫째 손가락의 지관절, 다른 네 손\n'
 '- 가락의 제1지관절(근위지관절)부터 심장에서 먼 쪽으로 손가락 뼈의 일부\n'
 '- 가 절단된 경우를 말하며, 뼈 단면이 불규칙해진 상태나 손가락 길이의\n'
 '- 단축 없이 골편만 떨어진 상태는 해당하지 않는다.\n'
 '- 7) ‘손가락에 뚜렷한 장해를 남긴 때’라 함은 첫째 손가락의 경우 중수지\n'
 '- 관절 또는 지관절의 굴신(굽히고 펴기)운동영역이 정상 운동영역의 1/2'),
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
 'indexing': {'chunk_id': 'chunk_000914',
              'chunk_char_len': 285,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
