from langchain_core.documents import Document

chunk = Document(
    page_content=('‘능력장애측정기준’ 상 6개 항목 중 2개<br>항목 이상에서 독립적 수행이 불가능하여 타인의 도움이 필요하고<br>GAF 60점 이하인 '
 '상태를 말한다.<br>사) ‘정신행동에 경미한 장해를 남긴 때’라 함은 장해판정 직전 1년 이<br>상 지속적인 정신건강의학과의 치료를 '
 "받았으며, 보건복지부고시<br>「장애정도판정기준」의 ‘능력장애측정기준’ 상 6개 항목 중 2개</p><br><p id='6' "
 "data-category='paragraph' style='font-size:16px'>154 KB 금쪽같은"),
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
 'indexing': {'chunk_id': 'chunk_001663',
              'chunk_char_len': 283,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
