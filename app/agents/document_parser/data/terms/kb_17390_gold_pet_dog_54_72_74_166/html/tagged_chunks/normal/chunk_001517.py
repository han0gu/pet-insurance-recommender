from langchain_core.documents import Document

chunk = Document(
    page_content=(". 장해의</h1><br><h1 id='212' style='font-size:14px'>평가기준</h1><br><p id='213' "
 "data-category='paragraph' style='font-size:14px'>1) 씹어먹는 기능의 장해는 윗니(상악치아)와 "
 '아랫니(하악치아)의 맞물림<br>(교합), 배열상태 및 아래턱의 개구(입벌리기)운동, 삼킴(연하)운동 등<br>에 따라 종합적으로 '
 '판단하여 결정한다.<br>2) ‘씹어먹는 기능에 심한 장해를 남긴 때’라 함은 심한 개구(입벌리기)운<br>동 제한이나 저작(씹기)운동'),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['dental', 'digestive']},
 'indexing': {'chunk_id': 'chunk_001517',
              'chunk_char_len': 298,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
