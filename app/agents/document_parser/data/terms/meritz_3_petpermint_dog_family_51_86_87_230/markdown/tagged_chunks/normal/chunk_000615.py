from langchain_core.documents import Document

chunk = Document(
    page_content=('- 나) 전실어증, 운동성실어증(브로카실어증)으로 의사\n'
 '- 소통이 불가한 경우\n'
 '- 8) “말하는 기능에 뚜렷한 장해를 남긴 때”라 함은\n'
 '- 아래의 경우 중 하나 이상에 해당되는 때를 말한\n'
 '- 다.\n'
 '- 가) 언어평가상 자음정확도가 50%미만인 경우\n'
 '- 나) 언어평가상 표현언어지수 25 미만인 경우\n'
 '- 9) “말하는 기능에 약간의 장해를 남긴 때”라 함은 아\n'
 '- 래의 경우 중 하나 이상에 해당되는 때를 말한다.\n'
 '- 가) 언어평가상 자음정확도가 75%미만인 경우\n'
 '- 나) 언어평가상 표현언어지수 65 미만인 경우'),
    metadata={'source_doc': {'total_pages': 180},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_3_petpermint_dog_family_51_86_87_230.pdf',
         'insurer_code': 'meritz',
         'product_code': '3',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000615',
              'chunk_char_len': 285,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
