from langchain_core.documents import Document

chunk = Document(
    page_content=('두 눈을 뜨고 10m 거리를 직선으로 걷다가 중간에 균형을 잡으려 멈 추어야 하는 경우 두 눈을 뜨고 10m 거리를 직선으로 걸을 때 '
 '중앙에서 60cm 이상 벗 어나는 경우 | 12 8\n'
 '2) 평형기능의 장해는 장해판정 직전 1년 이상 지속적인 치료 후 장해가 고착되었 을 때 판정하며, 뇌병변 여부, 전정기능 이상 및 '
 '장해상태를 평가하기 위해 아 래의 검사들을 기초로 한다.\n'
 '가) 뇌영상검사(CT, MRI) 나) 온도안진검사, 전기안진검사(또는 비디오안진검사) 등\n'
 '3. 코의 장해\n'
 '가. 장해의 분류\n'
 '장 해 의 분 류 | 지급률(%)'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 107,
         'page': 138},
 'term_type': 'special',
 'clause': {'clause_type': 'definition',
            'risk_domains': ['eye', 'head', 'other']},
 'indexing': {'chunk_id': 'chunk_000886',
              'chunk_char_len': 297,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
