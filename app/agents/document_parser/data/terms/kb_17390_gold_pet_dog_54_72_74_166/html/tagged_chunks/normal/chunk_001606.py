from langchain_core.documents import Document

chunk = Document(
    page_content=("흔들</p><br><p id='117' data-category='list' style='font-size:16px'>리거나 움직이는 "
 '것)이 있는 경우<br>다) 근전도 검사상 불완전한 손상(incomplete injury)소견이 있으면서 도<br>수근력검사(MMT)에서 '
 '근력이 3등급(fair)인 경우<br>11) 동요장해 평가 시에는 정상측과 환측을 비교하여 증가된 수치로 평가한다.</p><br><p '
 "id='118' data-category='paragraph' style='font-size:16px'>12) ‘가관절주 \ue045 이"),
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
 'indexing': {'chunk_id': 'chunk_001606',
              'chunk_char_len': 298,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
