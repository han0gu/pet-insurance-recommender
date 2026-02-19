from langchain_core.documents import Document

chunk = Document(
    page_content=('② 제1항 제4호의 사고증명서는 의료법 제3조(의료기관)에서 규정한 국내의 병원이나 의 원 또는 국외의 의료관련법에서 정한 의료기관에서 '
 '발급한 것이어야 합니다.\n'
 '<관련법규>\n'
 '[의료법 제3조(의료기관)]\n'
 '이 법에서 의료기관이라 함은 의료인이 공중 또는 특정 다수인을 위하여 의료·조산의 업을 행하는 곳을 말합니다. 의료기관은 종합병원· '
 '병원·치과병원·한방병원·요양병원·정신병원·의원·치과의원·한 의원 및 조산원으로 나누어집니다.\n'
 '제4조(보험금의 분담)'),
    metadata={'source_doc': {'total_pages': 107},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_4_pet_majestic_dog_33_49_52_139.pdf',
         'insurer_code': 'samsung',
         'product_code': '4',
         'product_name': '(장기)무배당 삼성화재 펫보험 위풍댕댕',
         'total_pages': 91,
         'page': 124},
 'term_type': 'special',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000784',
              'chunk_char_len': 251,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
