from langchain_core.documents import Document

chunk = Document(
    page_content=("136 -</p><p id='76' data-category='list'></p><h1 id='77' "
 "style='font-size:16px'>약 또는 보험료, 공제료 납입영수증에 장애인전용 보험, 공제로 표시된<br>보험, 공제의 "
 '보험료, 공제료를 말한다.<br>② 소득세법 제59조의 4 제1항 제2호에서 "대통령령으로 정하는 보험료"란</h1><br><h1 '
 "id='78' style='font-size:16px'>다음 각 호의 어느 하나에 해당하는 보험, 보증, 공제의 보험료, "
 "보증료,</h1><br><p id='79'"),
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
 'indexing': {'chunk_id': 'chunk_001411',
              'chunk_char_len': 294,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
