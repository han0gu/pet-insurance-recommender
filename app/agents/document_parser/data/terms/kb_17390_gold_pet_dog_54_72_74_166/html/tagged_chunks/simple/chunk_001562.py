from langchain_core.documents import Document

chunk = Document(
    page_content=('몸통) 한 개의 압박률이 40%이상인 경우 또는 한 운<br>동단위 내에 두 개 이상 척추체(척추뼈 몸통)의 압박골절로 각 '
 "척추</header><br><p id='58' data-category='list'></p><br><p id='59' "
 "data-category='paragraph' style='font-size:14px'>체(척추뼈 몸통)의 압박률의 합이 60% 이상일 "
 "때</p><br><p id='60' data-category='paragraph' style='font-size:14px'>11) 약간의 "
 '기형이란 다음 중'),
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
 'indexing': {'chunk_id': 'chunk_001562',
              'chunk_char_len': 297,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
