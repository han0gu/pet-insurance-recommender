from langchain_core.documents import Document

chunk = Document(
    page_content=('2) 평형기능의 장해는 장해판정 직전 1년 이상 지속적인\n'
 '치료 후 장해가 고착되었을 때 판정하며, 뇌병변 여\n'
 '부, 전정기능 이상 및 장해상태를 평가하기 위해 아\n'
 '래의 검사들을 기초로 한다.- 가) 뇌영상검사(CT, MRI)\n'
 '- 나) 온도안진검사, 전기안진검사(또는 비디오안진검사) 등\n'
 '205# 3. 코의 장해# 가. 장해의 분류| 장해의 분류 | 지급률 |\n'
 '| --- | --- |\n'
 '| 1) 코의 호흡기능을 완전히 잃었을 때 | 15 |\n'
 '| 2) 코의 후각기능을 완전히 잃었을 때 | 5 |'),
    metadata={'source_doc': {'total_pages': 180},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_3_petpermint_dog_family_51_86_87_230.pdf',
         'insurer_code': 'meritz',
         'product_code': '3',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['digestive', 'head']},
 'indexing': {'chunk_id': 'chunk_000605',
              'chunk_char_len': 274,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
