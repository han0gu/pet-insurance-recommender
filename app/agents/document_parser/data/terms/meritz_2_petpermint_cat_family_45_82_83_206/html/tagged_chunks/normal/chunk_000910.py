from langchain_core.documents import Document

chunk = Document(
    page_content=(". 신체부위</h1><br><p id='38' data-category='paragraph' "
 "style='font-size:16px'>“신체부위”라 함은 ① 눈 ② 귀 ③ 코 ④ 씹어먹거나<br>말하는 기능 ⑤ 외모 ⑥ "
 '척추(등뼈) ⑦ 체간골 ⑧ 팔 ⑨<br>다리 ⑩ 손가락 ⑪ 발가락 ⑫ 흉ㆍ복부 장기 및 비뇨생식<br>기 ⑬ 신경계ㆍ정신행동의 13개 '
 '부위를 말하며, 이를 각각<br>동일한 신체부위라 한다'),
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
 'indexing': {'chunk_id': 'chunk_000910',
              'chunk_char_len': 228,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
