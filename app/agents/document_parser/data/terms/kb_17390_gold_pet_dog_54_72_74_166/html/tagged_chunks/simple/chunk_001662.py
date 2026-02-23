from langchain_core.documents import Document

chunk = Document(
    page_content=('㉰ 적절한 대화기술 및<br>협조적인 대인관계, ㉱ 규칙적인 통원․약물 복용, ㉲ 소지품 및<br>금전관리나 적절한 구매행위, ㉳ '
 '대중교통이나 일반공공시설의<br>이용<br>바) ‘정신행동에 약간의 장해를 남긴 때’라 함은 장해판정 직전 1년 이</p><br><p '
 "id='5' data-category='list' style='font-size:16px'>상 지속적인 정신건강의학과의 치료를 받았으며, "
 '보건복지부고시<br>「장애정도판정기준」의 ‘능력장애측정기준’ 상 6개 항목 중 2개<br>항목 이상에서 독립적 수행이 불가능하여'),
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
 'indexing': {'chunk_id': 'chunk_001662',
              'chunk_char_len': 296,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
