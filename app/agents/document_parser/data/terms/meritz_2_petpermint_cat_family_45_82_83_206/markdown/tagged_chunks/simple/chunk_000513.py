from langchain_core.documents import Document

chunk = Document(
    page_content=('# 2. 신체부위“신체부위”라 함은 ① 눈 ② 귀 ③ 코 ④ 씹어먹거나\n'
 '말하는 기능 ⑤ 외모 ⑥ 척추(등뼈) ⑦ 체간골 ⑧ 팔 ⑨\n'
 '다리 ⑩ 손가락 ⑪ 발가락 ⑫ 흉ㆍ복부 장기 및 비뇨생식\n'
 '기 ⑬ 신경계ㆍ정신행동의 13개 부위를 말하며, 이를 각각\n'
 '동일한 신체부위라 한다. 다만, 좌ㆍ우의 눈, 귀, 팔,\n'
 '다리, 손가락, 발가락은 각각 다른 신체부위로 본다.# 3. 기타1) 하나의 장해가 관찰방법에 따라서 장해분류표상 2가지\n'
 '이상의 신체부위에서 장해로 평가되는 경우에는 그 중'),
    metadata={'source_doc': {'total_pages': 160},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_2_petpermint_cat_family_45_82_83_206.pdf',
         'insurer_code': 'meritz',
         'product_code': '2',
         'product_name': '무배당 펫퍼민트 Cat&Family보험 다이렉트2601',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other',
            'risk_domains': ['digestive', 'eye', 'head', 'urinary']},
 'indexing': {'chunk_id': 'chunk_000513',
              'chunk_char_len': 266,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
