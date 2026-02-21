from langchain_core.documents import Document

chunk = Document(
    page_content=('- 의 진단확정일부터 2년 이내로 하고, 보험기간이 10년 미만인 계약은 상해 발생\n'
 '- 일 또는 질병의 진단확정일부터 1년 이내)에 장해상태가 더 악화된 때에는 그 악\n'
 '- 화된 장해상태를 기준으로 장해지급률을 결정한다.\n'
 '# 2. 신체부위"신체부위" 라 함은 ① 눈 ② 귀 ③ 코 ④ 씹어먹거나 말하는 기능 ⑤ 외모 ⑥ 척추\n'
 '(등뼈) ⑦ 체간골 ⑧ 팔 ⑨ 다리 ⑩ 손가락 ⑪ 발가락 ⑫ 흉 · 복부 장기 및 비뇨생식기\n'
 '⑬ 신경계 · 정신행동의 13개 부위를 말하며, 이를 각각 동일한 신체부위라 한다. 다'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other',
            'risk_domains': ['digestive', 'eye', 'head', 'urinary']},
 'indexing': {'chunk_id': 'chunk_000730',
              'chunk_char_len': 281,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
