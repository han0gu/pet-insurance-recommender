from langchain_core.documents import Document

chunk = Document(
    page_content=('- 화된 때에는 그 악화된 장해상태를 기준으로 장해지급률을 결정한다.\n'
 '# 2. 신체부위‘신체부위’라 함은 ① 눈 ② 귀 ③ 코 ④ 씹어먹거나 말하는 기능 ⑤ 외모 ⑥ 척추(등뼈) ⑦ 체간골 ⑧ 팔 ⑨ 다리 '
 '⑩ 손가락 ⑪ 발가락 ⑫ 흉․복부장기 및 비뇨생식\n'
 '기 ⑬ 신경계․정신행동의 13개 부위를 말하며, 이를 각각 동일한 신체부위라 한다.'),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other',
            'risk_domains': ['digestive', 'eye', 'head', 'urinary']},
 'indexing': {'chunk_id': 'chunk_000833',
              'chunk_char_len': 189,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
