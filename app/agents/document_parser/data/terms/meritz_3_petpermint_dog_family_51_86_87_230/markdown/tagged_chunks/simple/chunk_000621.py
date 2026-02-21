from langchain_core.documents import Document

chunk = Document(
    page_content=('- 산하여 평가한다. 단, 길이가 5mm 미만의 반흔(흉\n'
 '- 터)은 합산대상에서 제외한다.\n'
 '- 5) 추상(추한 모습)이 얼굴과 머리 또는 목 부위에 걸\n'
 '- 쳐 있는 경우에는 머리 또는 목에 있는 흉터의 길이\n'
 '- 또는 면적의 1/2을 얼굴의 추상(추한 모습)으로 보\n'
 '- 아 산정한다.\n'
 '# 다. 뚜렷한 추상(추한 모습)# 1) 얼굴가) 손바닥 크기 1/2 이상의 추상(추한 모습)\n'
 '나) 길이 10cm 이상의 추상 반흔(추한 모습의 흉터)\n'
 '다) 지름 5cm 이상의 조직함몰'),
    metadata={'source_doc': {'total_pages': 180},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_3_petpermint_dog_family_51_86_87_230.pdf',
         'insurer_code': 'meritz',
         'product_code': '3',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['digestive', 'head']},
 'indexing': {'chunk_id': 'chunk_000621',
              'chunk_char_len': 261,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
