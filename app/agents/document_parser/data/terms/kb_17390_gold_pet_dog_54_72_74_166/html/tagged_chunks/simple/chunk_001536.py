from langchain_core.documents import Document

chunk = Document(
    page_content=(". 뚜렷한</h1><br><h1 id='15' style='font-size:14px'>1)</h1><br><p id='16' "
 "data-category='paragraph' style='font-size:14px'>추상(추한 모습)<br>얼굴<br>가) 손바닥 "
 '크기 1/2 이상의 추상(추한 모습)<br>나) 길이 10cm 이상의 추상 반흔(추한 모습의 흉터)<br>다) 지름 5cm 이상의 '
 "조직함몰</p><br><p id='17' data-category='list'></p><br><p id='18'"),
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
 'indexing': {'chunk_id': 'chunk_001536',
              'chunk_char_len': 281,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
