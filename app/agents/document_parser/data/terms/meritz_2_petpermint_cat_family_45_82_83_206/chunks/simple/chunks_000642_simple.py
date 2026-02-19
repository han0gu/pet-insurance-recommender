from langchain_core.documents import Document

chunk = Document(
    page_content=('경우 | 12\n'
 '러지는 두 눈을 뜨고 10미터 거리를 직선으로 걷 다가 중간에 균형을 잡으려 멈추어야 하 는 경우 두 눈을 뜨고 10m 거리를 직선으로 '
 '걸을 때 중앙에서 60cm 이상 벗어나는 경우 | 8\n'
 '2) 평형기능의 장해는 장해판정 직전 1년 이상 지속적인 치료 후 장해가 고착되었을 때 판정하며, 뇌병변 여 부, 전정기능 이상 및 '
 '장해상태를 평가하기 위해 아 래의 검사들을 기초로 한다.\n'
 '가) 뇌영상검사(CT, MRI) 나) 온도안진검사, 전기안진검사(또는 비디오안진검사) 등'),
    metadata={'source_doc': {'total_pages': 160},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_2_petpermint_cat_family_45_82_83_206.pdf',
         'insurer_code': 'meritz',
         'product_code': '2',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 160,
         'page': 180},
 'term_type': 'special',
 'clause': {'clause_type': 'exclusion',
            'risk_domains': ['eye', 'head', 'other']},
 'indexing': {'chunk_id': 'chunk_000642',
              'chunk_char_len': 269,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.85}},
)
