from langchain_core.documents import Document

chunk = Document(
    page_content=('대상이 되지<br>않으나, 선천적으로 영구치 결손이 있는 경우에는 유치의 결손을 후유<br>장해로 평가한다.<br>16) 가철성 '
 "보철물(신체의 일부에 붙였다 떼었다 할 수 있는 틀니 등)의 파</header><br><p id='1' "
 "data-category='paragraph' style='font-size:14px'>손은 후유장해의 대상이 되지 않는다.</p><p "
 "id='2' data-category='list'></p><br><h1 id='3' style='font-size:14px'>5"),
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
 'indexing': {'chunk_id': 'chunk_001529',
              'chunk_char_len': 277,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
