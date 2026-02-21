from langchain_core.documents import Document

chunk = Document(
    page_content=('- 상에 해당되는 때를 말한다.\n'
 '- 가) 언어평가상 자음정확도가 30%미만인 경우\n'
 '- 나) 전실어증, 운동성실어증(브로카실어증)으로 의사소통이 불가한 경우\n'
 '- 8) ‘말하는 기능에 뚜렷한 장해를 남긴 때’라 함은 아래의 경우 중 하나\n'
 '- 별\n'
 '- 이상에 해당되는 때를 말한다.\n'
 '- 표\n'
 '- 가) 언어평가상 자음정확도가 50%미만인 경우\n'
 '- 나) 언어평가상 표현언어지수 25 미만인 경우\n'
 '- 9) ‘말하는 기능에 약간의 장해를 남긴 때’라 함은 아래의 경우 중 하나\n'
 '- 이상에 해당되는 때를 말한다.'),
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
 'indexing': {'chunk_id': 'chunk_000863',
              'chunk_char_len': 276,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
