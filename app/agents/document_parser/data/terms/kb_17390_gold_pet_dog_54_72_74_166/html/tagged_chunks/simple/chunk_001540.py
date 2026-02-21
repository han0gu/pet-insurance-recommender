from langchain_core.documents import Document

chunk = Document(
    page_content=('지름 2cm 이상의 조직함몰<br>라) 코의 1/4 이상 결손<br>2) 머리<br>가) 손바닥 크기 1/2 이상의 반흔(흉터) 및 '
 '모발결손<br>나) 머리뼈의 손바닥 크기 1/2 이상의 손상 및 결손<br>3) 목<br>손바닥 크기 1/2 이상의 추상(추한 '
 "모습)</p><h1 id='27' style='font-size:14px'>마"),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['head']},
 'indexing': {'chunk_id': 'chunk_001540',
              'chunk_char_len': 188,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
