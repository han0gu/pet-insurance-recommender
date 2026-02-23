from langchain_core.documents import Document

chunk = Document(
    page_content=("말합니다.</td></tr></tbody></table><br><p id='128' data-category='paragraph' "
 "style='font-size:14px'>\uf000</p><br><p id='129' data-category='list' "
 "style='font-size:14px'>\uf000 제1항의 최대 수술길이란 하나의 독립된 반흔(흉터)의 최대 길이를 기준으로 "
 '하<br>며, 길이측정이 불가한 식피술(피부이식수술)등의 경우에는 반흔(흉터)을 벗어<br>나지 않는 범위에서 측정한 최대 직선길이로 '
 '합니다.<br>\uf000 제1항의'),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['digestive', 'skin']},
 'indexing': {'chunk_id': 'chunk_000441',
              'chunk_char_len': 294,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
