from langchain_core.documents import Document

chunk = Document(
    page_content=('② 제1항 제2호의 사고증명서는 의료법 제3조(의료기관)에서 규정한 국내의 병원이나 의\n'
 '원 또는 국외의 의료관련법에서 정한 의료기관에서 발급한 것이어야 합니다.<관련법규>[의료법 제3조(의료기관)]- 53 -이 법에서 '
 '의료기관이라 함은 의료인이 공중 또는 특정 다수인을 위하여 의료·조산의 업을 행하는\n'
 '곳을 말합니다. 의료기관은 종합병원·병원·치과병원·한방병원·요양병원·정신병원·의원·치과의원·한'),
    metadata={'source_doc': {'total_pages': 107},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_4_pet_majestic_dog_33_49_52_139.pdf',
         'insurer_code': 'samsung',
         'product_code': '4',
         'product_name': '(장기)무배당 삼성화재 펫보험 위풍댕댕',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000168',
              'chunk_char_len': 221,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
