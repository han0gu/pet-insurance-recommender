from langchain_core.documents import Document

chunk = Document(
    page_content=('- 큼, 즉 이 사고와의 관여도를 산정하여 평가한다.\n'
 '- 4) 추간판탈출증으로 인한 신경 장해는 수술 또는 시술(비수술적 치료) 후 6개월\n'
 '- 이상 지난 후에 평가한다.\n'
 '- 5) 신경학적 검사상 나타난 저린감이나 방사통 등 신경자극증상의 원인으로 CT,\n'
 '- MRI 등 영상검사에서 추간판탈출증이 확인된 경우를 추간판탈출증으로 진단\n'
 '- 하며, 수술 여부에 관계없이 운동장해 및 기형장해로 평가하지 않는다.\n'
 '- 6) 심한 운동장해란 다음 중 어느 하나에 해당하는 경우를 말한다.'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['digestive', 'head']},
 'indexing': {'chunk_id': 'chunk_000772',
              'chunk_char_len': 265,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
