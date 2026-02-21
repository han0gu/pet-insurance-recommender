from langchain_core.documents import Document

chunk = Document(
    page_content=('. 기본공제대상자를 피보험자로 하는 대통령령으로 정하는 보험료(제1호에<br>따른 장애인전용보장성보험료는 제외한다)<br>소득세법 시행령 '
 '제118조의 4(보험료의 세액공제)<br>① 소득세법 제59조의 4 제1항 제1호에서 "대통령령으로 정하는 '
 '장애인전용<br>보장성보험료"란 제2항 각 호에 해당하는 보험, 공제로서 보험, 공제 계</p><p id=\'75\' '
 "data-category='paragraph' style='font-size:16px'>- 136 -</p><p id='76' "
 "data-category='list'></p><h1"),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_001410',
              'chunk_char_len': 300,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
