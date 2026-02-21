from langchain_core.documents import Document

chunk = Document(
    page_content=('투시검사)상 연하장애가 있고, 유동식 섭취<br>시 흡인이 발생하고 연식 외에는 섭취가 불가능한 상태<br>4) ‘씹어먹는 기능에 약간의 '
 "장해를 남긴 때’라 함은 아래의 경우 중 하나</p><br><p id='216' "
 "data-category='list'></p><br><footer id='217' style='font-size:14px'>이상에 "
 '해당되는 때를 말한다.<br>가) 약간의 개구(입벌리기)운동 제한 또는 약간의 저작(씹기)운동 제한<br>공<br>으로 부드러운 '
 '고형식(밥, 빵 등)만 섭취 가능한'),
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
 'indexing': {'chunk_id': 'chunk_001520',
              'chunk_char_len': 286,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
