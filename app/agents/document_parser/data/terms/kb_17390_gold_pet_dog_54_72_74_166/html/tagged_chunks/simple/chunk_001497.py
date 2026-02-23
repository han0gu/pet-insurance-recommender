from langchain_core.documents import Document

chunk = Document(
    page_content=('. ㆍ<br>3) ‘심한 장해를 남긴 때’라 함은 순음청력검사 결과 평균순음역치가 80dB 규정<br>이상인 경우에 해당되어, 귀에다 '
 '대고 말하지 않고는 큰 소리를 알아듣<br>지 못하는 경우를 말한다.<br>4) ‘약간의 장해를 남긴 때’라 함은 순음청력검사 결과 '
 '평균순음역치가<br>70dB 이상인 경우에 해당되어, 50cm 이상의 거리에서는 보통의 말소리<br>를 알아듣지 못하는 경우를 '
 '말한다.<br>5) 순음청력검사를 실시하기 곤란하거나(청력의 감소가 의심되지만 의사소<br>통이 되지 않는 경우, 만 3세 미만의 소아 '
 '포함)'),
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
 'indexing': {'chunk_id': 'chunk_001497',
              'chunk_char_len': 299,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
