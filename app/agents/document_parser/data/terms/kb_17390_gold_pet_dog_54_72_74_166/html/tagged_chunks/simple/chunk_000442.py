from langchain_core.documents import Document

chunk = Document(
    page_content=('벗어<br>나지 않는 범위에서 측정한 최대 직선길이로 합니다.<br>\uf000 제1항의 "성형수술" 은 피보험자가 사고발생시점에 '
 "만15세 미만일 경우 부득이</p><br><p id='130' data-category='paragraph' "
 "style='font-size:16px'>- 78 -</p><p id='131' data-category='paragraph' "
 "style='font-size:16px'>사고일로부터 2년이 지난 후에 성형수술이 가능하다는 진단을 받은 경우에는 그</p><br><h1 "
 "id='132'"),
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
 'indexing': {'chunk_id': 'chunk_000442',
              'chunk_char_len': 285,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
