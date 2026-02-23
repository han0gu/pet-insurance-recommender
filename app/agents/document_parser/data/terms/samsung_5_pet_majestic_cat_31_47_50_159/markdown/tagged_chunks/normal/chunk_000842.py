from langchain_core.documents import Document

chunk = Document(
    page_content=('- 을 말한다.\n'
 '- 나) 치매의 장해평가는 임상적인 증상 뿐 아니라 뇌영상검사(CT 및 MRI,\n'
 '- SPECT 등)를 기초로 진단되어져야 하며, 18개월 이상 지속적인 치료 후\n'
 '- 평가한다. 다만, 진단시점에 이미 극심한 치매 또는 심한 치매로 진행된 경\n'
 '- 우에는 6개월간 지속적인 치료 후 평가한다.\n'
 '- 다) 치매의 장해평가는 전문의(정신건강의학과, 신경과)에 의한 임상치매척도\n'
 '- (한국판 Expanded Clinical Dementia Rating) 검사결과에 따른다.\n'
 '- 4) 뇌전증'),
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
 'indexing': {'chunk_id': 'chunk_000842',
              'chunk_char_len': 275,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
