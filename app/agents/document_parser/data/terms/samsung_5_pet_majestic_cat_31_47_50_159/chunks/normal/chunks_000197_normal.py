from langchain_core.documents import Document

chunk = Document(
    page_content=('이 법에서 의료기관이라 함은 의료인이 공중 또는 특정 다수인을 위하여 의료·조산의 업을 행하는 곳을 말합니다. 의료기관은 '
 '종합병원·병원·치과병원·한방병원·요양병원·정신병원·의원·치과의원·한 의원 및 조산원으로 나누어집니다.\n'
 '제10조 (보험금 등의 지급절차)'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 107,
         'page': 52},
 'term_type': 'special',
 'clause': {'clause_type': 'claim', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000197',
              'chunk_char_len': 143,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.55}},
)
