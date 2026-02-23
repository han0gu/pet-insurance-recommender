from langchain_core.documents import Document

chunk = Document(
    page_content=("몸통)에 골절 또는 탈구로 2개의 척추체(척추뼈 몸통)를 유합(아</p><br><h1 id='46' "
 "style='font-size:16px'>물어 붙음) 또는 고정한 상태</h1><br><p id='47' "
 "data-category='paragraph' style='font-size:16px'>9) 심한 기형이란 다음 중 어느 하나에 해당하는 "
 "경우를 말한다.</p><br><p id='48' data-category='paragraph' "
 "style='font-size:16px'>가) 척추(등뼈)의 골절 또는 탈구 등으로 35° 이상의"),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['joint']},
 'indexing': {'chunk_id': 'chunk_001557',
              'chunk_char_len': 299,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
