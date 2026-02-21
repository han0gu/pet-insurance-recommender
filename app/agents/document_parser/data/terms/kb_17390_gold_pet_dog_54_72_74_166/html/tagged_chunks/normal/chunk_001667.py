from langchain_core.documents import Document

chunk = Document(
    page_content=("각종 신경증 및 각종 인격장애는 보상의 대상이 되지<br>않는다.</p><p id='8' data-category='paragraph' "
 "style='font-size:16px'>3) 치매</p><br><p id='9' data-category='list'></p><p "
 "id='10' data-category='list' style='font-size:16px'>가) “치매”라 함은 정상적으로 성숙한 뇌가 "
 '질병이나 외상 후 기질성<br>손상으로 파괴되어 한번 획득한 지적기능이 지속적 또는 전반적으로<br>저하되는 것을'),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['digestive', 'head']},
 'indexing': {'chunk_id': 'chunk_001667',
              'chunk_char_len': 290,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
