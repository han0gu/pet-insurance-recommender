from langchain_core.documents import Document

chunk = Document(
    page_content=(". 기타</h1><br><p id='163' data-category='paragraph' style='font-size:14px'>1) "
 "하나의 장해가 관찰 방법에 따라서 장해분류표상 2가지 이상의 신체부위에서</p><br><p id='164' "
 "data-category='list' style='font-size:14px'>장해로 평가되는 경우에는 그 중 높은 지급률을 "
 '적용한다.<br>2) 동일한 신체부위에 2가지 이상의 장해가 발생한 경우에는 합산하지 않고 그<br>중 높은 지급률을 적용함을 원칙으로 '
 '한다'),
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
 'indexing': {'chunk_id': 'chunk_001473',
              'chunk_char_len': 285,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
