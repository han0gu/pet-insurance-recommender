from langchain_core.documents import Document

chunk = Document(
    page_content=('| 제1항에서 정한 안면부란 이마를 포함하여 | 목까지의 얼굴부분을 말합니다. |\n'
 '\uf000- \uf000 제1항의 최대 수술길이란 하나의 독립된 반흔(흉터)의 최대 길이를 기준으로 하\n'
 '- 며, 길이측정이 불가한 식피술(피부이식수술)등의 경우에는 반흔(흉터)을 벗어\n'
 '- 나지 않는 범위에서 측정한 최대 직선길이로 합니다.\n'
 '- \uf000 제1항의 "성형수술" 은 피보험자가 사고발생시점에 만15세 미만일 경우 부득이'),
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
 'indexing': {'chunk_id': 'chunk_000286',
              'chunk_char_len': 218,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
