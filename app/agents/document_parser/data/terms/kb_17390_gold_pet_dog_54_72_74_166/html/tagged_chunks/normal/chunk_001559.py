from langchain_core.documents import Document

chunk = Document(
    page_content=('이상 척추체(척추뼈 몸통)의 압박골절로 각 척추<br>체(척추뼈 몸통)의 압박률의 합이 90% 이상일 때<br>뚜렷한 기형이란 다음 중 '
 "어느 하나에 해당하는 경우를 말한다.</p><br><p id='50' data-category='list'></p><br><p "
 "id='51' data-category='paragraph' style='font-size:16px'>10)</p><p id='52' "
 "data-category='paragraph' style='font-size:16px'>KB 금쪽같은 "
 '펫보험(강아지)(무배당)(26.01)'),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_001559',
              'chunk_char_len': 299,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
