from langchain_core.documents import Document

chunk = Document(
    page_content=('하나의 운동단위로 보며, 하<br>나의 운동단위 내에서 여러 개의 척추체(척추뼈 몸통)에 압박골절이<br>발생한 경우에는 각 '
 "척추체(척추뼈 몸통)의 압박률을 합산하고, 두</p><br><p id='39' data-category='list' "
 "style='font-size:16px'>개 이상의 운동단위에서 장해가 발생한 경우에는 그 중 가장 높은<br>지급률을 "
 '적용한다.<br>공<br>3) 척추(등뼈)의 장해는 퇴행성 기왕증 병변과 사고가 그 증상을 악화시킨<br>통<br>부분만큼, 즉 이 '
 '사고와의 관여도를 산정하여'),
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
 'indexing': {'chunk_id': 'chunk_001551',
              'chunk_char_len': 290,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
